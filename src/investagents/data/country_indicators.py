"""Indicateurs pays-année (WDI, CPIA, WGI) rattachés aux projets, connus à l'approbation.

Sources (vérifiées par appels réels le 2026-11-15) :
- WDI, API Indicators v2, source 2 (`/v2/country/all/indicator/{code}?source=2`, pages de
  1 000) : PIB/hab. constant (NY.GDP.PCAP.KD), croissance (NY.GDP.MKTP.KD.ZG), inflation
  (FP.CPI.TOTL.ZG), population (SP.POP.TOTL), aide reçue en % du RNB (DT.ODA.ODAT.GN.ZS), et le
  CPIA « IDA resource allocation index » (IQ.CPA.IRAI.XQ, 1 à 6, publié à partir de 2005) ;
- WGI : fichier complet du DataBank `WGI_CSV.zip` (les sources 3 et 31 de l'API renvoyaient 503
  ou expiraient ce jour-là). Six estimations GOV_WGI_*.EST, 1996–2025 (tous les deux ans
  jusqu'en 2002).

Règle ex ante : pour un projet approuvé en exercice N, on prend la DERNIÈRE valeur disponible
d'une année ≤ N − 2 (une valeur de l'année N−1 n'est en général publiée qu'au milieu de l'année
N, donc parfois après l'approbation), et au plus 4 ans en arrière. Sinon : manquant.

Limite connue : ce sont les valeurs ACTUELLES de ces bases (révisées depuis), pas les valeurs
telles que publiées à l'époque.

Pays du projet : `country_code` (ISO2) des relevés de prêts IDA/IBRD (`data/raw/statement_*`),
le plus fréquent parmi les prêts du projet. Projets régionaux ou hors IDA/IBRD : manquant.

Lancer : `uv run python -m investagents.data.country_indicators`
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from investagents.data.http import get_json

API = "https://api.worldbank.org/v2"
WDI_CODES = {
    "NY.GDP.PCAP.KD": "gdp_pc",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "SP.POP.TOTL": "population",
    "DT.ODA.ODAT.GN.ZS": "oda_gni",
    "IQ.CPA.IRAI.XQ": "cpia",
}
WGI_CODES = {
    "GOV_WGI_CC.EST": "wgi_cc",
    "GOV_WGI_GE.EST": "wgi_ge",
    "GOV_WGI_PV.EST": "wgi_pv",
    "GOV_WGI_RL.EST": "wgi_rl",
    "GOV_WGI_RQ.EST": "wgi_rq",
    "GOV_WGI_VA.EST": "wgi_va",
}
LAG_YEARS = 2
MAX_STALENESS = 4

WGI_ZIP = Path("data/raw/wgi/WGI_CSV.zip")
WDI_RAW = Path("data/raw/wdi_long.parquet")
LONG_PATH = Path("data/processed/country_year_indicators.parquet")
PROJECT_COUNTRY_PATH = Path("data/processed/project_country.parquet")


YEAR_WINDOWS = [(y, y + 4) for y in range(1960, 2026, 5)]  # requêtes par tranches de 5 ans


def fetch_wdi(code: str, get=get_json) -> tuple[pd.DataFrame, dict[str, str]]:
    """Toutes les années, tous les pays, pour un indicateur WDI ; + table ISO2 -> ISO3.

    Une requête 1960:2025 d'un coup expirait côté serveur (observé) ; par tranches de 5 ans,
    chaque page répond en quelques secondes.
    """
    rows, iso_map = [], {}
    for start, end in YEAR_WINDOWS:
        page = 1
        while True:
            r = get(
                f"{API}/country/all/indicator/{code}",
                {
                    "format": "json",
                    "source": 2,
                    "date": f"{start}:{end}",
                    "per_page": 1000,
                    "page": page,
                },
            )
            meta, data = r[0], (r[1] if len(r) > 1 else None) or []
            for x in data:
                iso_map[x["country"]["id"]] = x["countryiso3code"]
                if x["value"] is not None and x["countryiso3code"]:
                    rows.append(
                        {"iso3": x["countryiso3code"], "year": int(x["date"]), "value": x["value"]}
                    )
            if page >= int(meta.get("pages") or 1):
                break
            page += 1
    df = pd.DataFrame(rows, columns=["iso3", "year", "value"]).drop_duplicates(["iso3", "year"])
    return df, iso_map


def read_wgi(zip_path: Path = WGI_ZIP) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(z.open("WGICSV.csv"))
    df = df[df["Indicator Code"].isin(WGI_CODES)]
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(
        id_vars=["Country Code", "Indicator Code"], value_vars=year_cols, var_name="year"
    ).dropna(subset=["value"])
    return pd.DataFrame(
        {
            "iso3": long["Country Code"],
            "year": long["year"].astype(int),
            "indicator": long["Indicator Code"].map(WGI_CODES),
            "value": long["value"].astype(float),
        }
    )


def project_countries(statements: list[pd.DataFrame], iso2_to_iso3: dict[str, str]) -> pd.DataFrame:
    """Un pays (ISO3) par projet : le code le plus fréquent parmi ses prêts."""
    s = pd.concat([d[["project_id", "country_code"]] for d in statements], ignore_index=True)
    # Codes régionaux observés : « 3E », « 6R », « E4 », « P3 »... (toujours un chiffre).
    s = s[s["country_code"].astype(str).str.fullmatch(r"[A-Z]{2}")]
    top = s.groupby("project_id")["country_code"].agg(lambda c: c.value_counts().index[0])
    out = top.rename("iso2").reset_index()
    out["iso3"] = out["iso2"].map(iso2_to_iso3)
    return out


def as_of_join(projects: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque projet et indicateur : dernière valeur d'année ≤ N−2, au plus 4 ans d'âge.

    `projects` : project_id, iso3, approval_fy. `long` : iso3, year, indicator, value.
    Renvoie une ligne par projet, une colonne par indicateur.
    """
    p = projects.dropna(subset=["iso3"]).copy()
    p["target_year"] = p["approval_fy"].astype(int) - LAG_YEARS
    out = projects[["project_id"]].copy()
    for ind, g in long.groupby("indicator"):
        g = g.sort_values("year")[["iso3", "year", "value"]]
        left = p[["project_id", "iso3", "target_year"]].sort_values("target_year")
        m = pd.merge_asof(
            left,
            g.rename(columns={"year": "obs_year"}),
            left_on="target_year",
            right_on="obs_year",
            by="iso3",
            direction="backward",
        )
        m.loc[m["target_year"] - m["obs_year"] > MAX_STALENESS, "value"] = np.nan
        out = out.merge(m[["project_id", "value"]].rename(columns={"value": ind}), how="left")
    return out


def main() -> None:
    frames, iso_map = [], {}
    if WDI_RAW.exists():
        wdi = pd.read_parquet(WDI_RAW)
        iso_map = dict(pd.read_parquet(WDI_RAW.with_name("iso2_iso3.parquet")).values)
        print(f"Cache utilisé : {WDI_RAW}")
    else:
        cache = WDI_RAW.parent / "wdi_cache"
        cache.mkdir(parents=True, exist_ok=True)
        for code, name in WDI_CODES.items():
            part, part_iso = cache / f"{name}.parquet", cache / f"{name}_iso.parquet"
            if part.exists():  # reprise après une coupure réseau
                df, m = pd.read_parquet(part), dict(pd.read_parquet(part_iso).values)
            else:
                df, m = fetch_wdi(code)
                df.to_parquet(part, index=False)
                pd.DataFrame(list(m.items()), columns=["iso2", "iso3"]).to_parquet(part_iso)
            iso_map.update(m)
            df["indicator"] = name
            frames.append(df)
            print(f"WDI {code} : {len(df)} valeurs", flush=True)
        wdi = pd.concat(frames, ignore_index=True)
        wdi.to_parquet(WDI_RAW, index=False)
        pd.DataFrame(list(iso_map.items()), columns=["iso2", "iso3"]).to_parquet(
            WDI_RAW.with_name("iso2_iso3.parquet"), index=False
        )
    long = pd.concat([wdi, read_wgi()], ignore_index=True)
    LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(LONG_PATH, index=False)

    statements = [pd.read_parquet(f"data/raw/statement_{s}_raw.parquet") for s in ("ida", "ibrd")]
    pc = project_countries(statements, iso_map)
    pc.to_parquet(PROJECT_COUNTRY_PATH, index=False)
    print(f"Indicateurs : {long['indicator'].nunique()}")
    print(f"Projets avec pays : {pc['iso3'].notna().sum()}")


if __name__ == "__main__":
    main()
