"""Contrôles descriptifs des variables de secteur et des champs API non utilisés (limite).

Aucun modèle, aucune issue : seulement des comptes sur les variables d'entrée.
1. secteur (`global_practice`) manquant par période d'approbation ;
2. nomenclature en 2014 : mêmes libellés avant / après ? plus grand changement de part d'un
   secteur entre 2013 et 2014 ;
3. WGI manquant à l'entraînement (groupe testé puis écarté en phase 2) ;
4. champs de l'API Projects disponibles mais non utilisés (catégorie environnementale, secteurs) :
   un appel réel sur un projet (P101171), liste des champs présents.

Écrit experiments/data_checks/sectors.json, lu par report/metrics.py (la démo et la CI n'ont pas
les données brutes).

Lancer : `uv run python -m investagents.data.sector_checks`
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from investagents.data.http import get_json
from investagents.ml.run_phase2 import load_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/data_checks/sectors.json"
API = "https://search.worldbank.org/api/v2/projects"
PROJECT = "P101171"
UNUSED = ["envassesmentcategorycode", "sector1", "sector2", "sector3", "sectorcode"]


def main() -> None:
    t, _, _ = load_data()
    gp = t.global_practice.astype("string").str.strip().replace("", pd.NA)
    old = t.approval_fy <= 1995
    out = {
        "n_le1995": int(old.sum()),
        "sector_missing_le1995": float(gp[old].isna().mean()),
        "sector_missing_after1995": float(gp[~old].isna().mean()),
    }
    before = gp[(t.approval_fy >= 2009) & (t.approval_fy <= 2013)].dropna()
    after = gp[(t.approval_fy >= 2014) & (t.approval_fy <= 2017)].dropna()
    out["labels_2009_2013"] = int(before.nunique())
    out["labels_2014_2017"] = int(after.nunique())
    out["labels_only_after_2014"] = sorted(set(after) - set(before))
    out["labels_only_before_2014"] = sorted(set(before) - set(after))
    s13 = gp[t.approval_fy == 2013].value_counts(normalize=True)
    s14 = gp[t.approval_fy == 2014].value_counts(normalize=True)
    jump = (s14.reindex(s13.index.union(s14.index), fill_value=0)
            - s13.reindex(s13.index.union(s14.index), fill_value=0)).abs()  # fmt: skip
    out["max_share_change_2013_2014"] = float(jump.max())
    out["max_share_change_sector"] = str(jump.idxmax())
    tr = t.split == "train"
    out["wgi_missing_train"] = float(t.loc[tr, "wgi_ge"].isna().mean())
    resp = get_json(API, params={"format": "json", "id": PROJECT})
    fields = sorted(next(iter(resp["projects"].values())).keys())
    out["api_check"] = {"project": PROJECT, "n_fields": len(fields),
                        "unused_fields_present": [f for f in UNUSED if f in fields],
                        "has_preparation_start_date": any("concept" in f or "ident" in f
                                                          for f in fields)}  # fmt: skip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
