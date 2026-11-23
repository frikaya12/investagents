"""Tous les chiffres du rapport final, de la vitrine et de la démo, lus dans les résultats.

Règle : aucun chiffre n'est tapé à la main dans le rapport ou la démo. Chaque chiffre est lu ici,
dans un JSON de `experiments/` (écrit par le script qui a fait l'expérience). Les rares chiffres qui
n'existent que dans un rapport généré (`reports/*.md`, écrit par un script) sont lus avec
`_from_report`, qui vérifie que le texte est bien présent dans ce rapport.

`load()` renvoie un objet `Metrics` :
- `m.txt[clé]` : le chiffre déjà mis en forme en français (« 0,297 [0,245 ; 0,369] ») ;
- `m.raw[clé]` : la valeur brute (float, dict) pour les graphiques ;
- `m.source[clé]` : le fichier d'où vient le chiffre.

Aucun calcul de modèle ici : seulement de la lecture et de la mise en forme.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MINUS = "−"  # vrai signe moins, comme dans le rapport
NBSP = "\u00a0"  # espace insécable : « 8 260 » et « 85,3 % » ne se coupent pas en fin de ligne

CRITERIA_FR = {
    "relevance": "Pertinence",
    "economic_efficiency": "Efficience économique",
    "feasibility": "Faisabilité",
    "risk": "Risque",
    "impact_sustainability": "Impact et durabilité",
}


# ---------- mise en forme française ----------


def num(x: float, nd: int = 3, signed: bool = False) -> str:
    """0.2975 -> « 0,297 » ; signé : « +0,039 » ou « −0,031 »."""
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    if signed:
        return ("+" if x >= 0 else MINUS) + s if round(x, nd) != 0 else s
    return (MINUS if x < 0 and round(x, nd) != 0 else "") + s


def ci(d: dict, nd: int = 3, signed: bool = False) -> str:
    """{'value', 'ci_low', 'ci_high'} -> « 0,297 [0,245 ; 0,369] »."""
    return f"{num(d['value'], nd, signed)} [{num(d['ci_low'], nd)} ; {num(d['ci_high'], nd)}]"


def pct(x: float, nd: int = 1) -> str:
    """0.8533 -> « 85,3 % »."""
    return f"{x * 100:.{nd}f}".replace(".", ",") + NBSP + "%"


def pct_ci(v: float, lo: float, hi: float) -> str:
    """« 85,3 % [79,3 ; 90,7] »."""
    lo_s, hi_s = (f"{x * 100:.1f}".replace(".", ",") for x in (lo, hi))
    return f"{pct(v)} [{lo_s} ; {hi_s}]"


def integer(n: int) -> str:
    """8260 -> « 8 260 » (espace des milliers)."""
    return f"{int(n):,}".replace(",", NBSP)


def pval(p: float, n_draws: int = 1000) -> str:
    """Aucun tirage meilleur sur 1 000 -> « p ≤ 0,001 » (plus petite valeur atteignable)."""
    if p <= 1 / n_draws:
        return "p ≤ " + num(1 / n_draws, 3)
    return "p = " + num(p, 3)


# ---------- lecture ----------


def _json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _from_report(rel: str, text: str) -> str:
    """Chiffre présent seulement dans un rapport généré : on vérifie qu'il y est bien écrit."""
    content = (ROOT / rel).read_text(encoding="utf-8")
    if text not in content:
        raise ValueError(f"« {text} » introuvable dans {rel}")
    return text


@dataclass
class Metrics:
    txt: dict[str, str] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)
    source: dict[str, str] = field(default_factory=dict)

    def put(self, key: str, text: str, raw: object, src: str) -> None:
        if key in self.txt:
            raise KeyError(f"clé en double : {key}")
        self.txt[key], self.raw[key], self.source[key] = text, raw, src


P2 = "experiments/phase2/results.json"
BB = "experiments/blocB/results.json"
OBJ = "experiments/blocA/objective_test.json"
BC = "experiments/blocC/models_results.json"
BCF = "experiments/blocC/models_frozen.json"
M7 = "experiments/blocC/m7_results.json"
M7F = "experiments/blocC/m7_frozen.json"
COM = "experiments/blocC/committee_summary.json"
FT = "experiments/blocC/frontier_test.json"
FV = "experiments/blocC/frontier_valid.json"
D6 = "experiments/blocD/m6_results.json"
D6F = "experiments/blocD/m6_frozen.json"
DP = "experiments/blocD/kaggle_pilot/summary.json"
DF = "experiments/blocD/kaggle_full1/summary.json"
DM = "experiments/blocD/mapping_check.json"
V = "experiments/vault/results.json"


@lru_cache(maxsize=1)
def load() -> Metrics:  # noqa: PLR0915 (une longue liste de lectures, volontairement à plat)
    m = Metrics()

    # --- données ---
    feas = "reports/faisabilite.md"
    m.put("n_exploitable", _from_report(feas, "8 260"), 8260, feas)
    m.put("n_fail_exploitable", _from_report(feas, "2 210"), 2210, feas)
    p2 = _json(P2)
    for s in ("train", "valid", "test"):
        c = p2["counts"][s]
        m.put(f"n_{s}", integer(c["n"]), c["n"], P2)
        m.put(f"fail_{s}", pct(c["failure_rate"], 0), c["failure_rate"], P2)

    # --- phase 2 : M0, M1 ---
    tf = p2["test_full"]
    m.put("m0_aucpr", num(tf["M0"]["auc_pr"]["value"]), tf["M0"]["auc_pr"]["value"], P2)
    m.put("m1_aucpr", ci(tf["M1"]["auc_pr"]), tf["M1"]["auc_pr"], P2)
    m.put("m1_aucroc", ci(tf["M1"]["auc_roc"]), tf["M1"]["auc_roc"], P2)
    d = p2["diff_full"]["M1 - M0"]["auc_pr"]
    m.put("m1_minus_m0", ci(d, signed=True), d, P2)
    m.put("m1_perm_p", pval(p2["chance"]["M1_auc_pr"]["p_value"], 2000),
          p2["chance"]["M1_auc_pr"]["p_value"], P2)  # fmt: skip
    m.put("m1_valid_aucpr", num(p2["frozen"]["m1"]["valid_auc_pr"]),
          p2["frozen"]["m1"]["valid_auc_pr"], P2)  # fmt: skip
    m.put("m1t_valid_aucpr", num(p2["frozen"]["m1_text"]["valid_auc_pr"]),
          p2["frozen"]["m1_text"]["valid_auc_pr"], P2)  # fmt: skip
    m.put("m1t_train_aucpr", num(p2["m1_text_diagnostic"]["train"]["auc_pr"]),
          p2["m1_text_diagnostic"]["train"]["auc_pr"], P2)  # fmt: skip
    m.put("m1_features", p2["frozen"]["m1"]["feature_set"], p2["frozen"]["m1"]["feature_set"], P2)
    for s in ("train", "valid", "test"):
        n = p2["m1_text_counts"][s]
        m.put(f"n_text_{s}", integer(n), n, P2)
    n_all = sum(p2["counts"][s]["n"] for s in ("train", "valid", "test"))
    m.put("n_rated", integer(n_all), n_all, P2)

    # --- bloc B : M1-texte figé, calibration, portefeuille ---
    bb = _json(BB)
    mod = bb["models"]
    m.put("n_test_text", integer(bb["n_test"]), bb["n_test"], BB)
    m.put("base_test_text", num(bb["test_failure_rate"]), bb["test_failure_rate"], BB)
    m.put("m1t_aucpr", ci(mod["m1_text"]["auc_pr"]), mod["m1_text"]["auc_pr"], BB)
    m.put("m1t_aucroc", ci(mod["m1_text"]["auc_roc"]), mod["m1_text"]["auc_roc"], BB)
    m.put("m1_on_text_aucpr", ci(mod["m1"]["auc_pr"]), mod["m1"]["auc_pr"], BB)
    d = bb["diffs"]["chosen - m1"]["auc_pr"]
    m.put("m1t_minus_m1", ci(d, signed=True), d, BB)
    d = bb["diffs"]["tfidf_masked - m1_text"]["auc_pr"]
    m.put("masked_minus_m1t", ci(d, signed=True), d, BB)
    c = mod["m1_text"]
    m.put("ece_before", num(c["ece_before"]), c["ece_before"], BB)
    m.put("ece_after", num(c["ece_after"]), c["ece_after"], BB)
    m.put("brier_before", num(c["brier_before"]), c["brier_before"], BB)
    m.put("brier_after", num(c["brier"]["value"]), c["brier"]["value"], BB)
    by_budget = {r["budget"]: r for r in bb["portfolio"]}
    m0_ctrl = {r["budget"]: r for r in bb["portfolio_control"]["m0"]}
    for b, tag in ((0.25, "25"), (0.5, "50"), (0.1, "10")):
        r = by_budget[b]
        m.put(f"pf{tag}", pct_ci(r["count_share_satisfactory"], *r["count_share_ci"]), r, BB)
        m.put(f"pf{tag}_share", pct(r["count_share_satisfactory"]), r["count_share_satisfactory"],
              BB)  # fmt: skip
        m.put(f"pf{tag}_random", pct(r["random_count"]), r["random_count"], BB)
        m.put(f"pf{tag}_p", pval(r["p_count_vs_random"]), r["p_count_vs_random"], BB)
        m.put(f"pf{tag}_m0", pct(m0_ctrl[b]["count_share_satisfactory"]), m0_ctrl[b], BB)
        m.put(f"pf{tag}_m0_p", pval(m0_ctrl[b]["p_count_vs_random"]),
              m0_ctrl[b]["p_count_vs_random"], BB)  # fmt: skip
    m.put("pf_n_pool", integer(by_budget[0.25]["n_pool"]), by_budget[0.25]["n_pool"], BB)
    m.put("pf25_n", integer(by_budget[0.25]["n_projects"]), by_budget[0.25]["n_projects"], BB)
    obj = _json(OBJ)
    r25 = next(r for r in obj["results"] if r["budget"] == 0.25)
    rnd = r25["random"]["count_share_satisfactory"]
    m.put("pf25_random_band", f"{pct(rnd['p2_5'])} à {pct(rnd['p97_5'])}", rnd, OBJ)
    m.put("pf25_initial_p", pct(r25["initial_p"]["count_share_satisfactory"]),
          r25["initial_p"], OBJ)  # fmt: skip
    m.put("pf25_initial_p_p", pval(r25["initial_p"]["p_value_count_vs_random"]),
          r25["initial_p"]["p_value_count_vs_random"], OBJ)  # fmt: skip

    # --- coffre-fort ---
    v = _json(V)["all"]
    m.put("vault_n", integer(v["n"]), v["n"], V)
    m.put("vault_fail", pct(v["failure_rate"]), v["failure_rate"], V)
    m.put("vault_m0_aucpr", num(v["failure_rate"]), v["failure_rate"], V)
    m.put("vault_m1_aucpr", ci(v["models"]["M1"]["ci"]["auc_pr"]),
          v["models"]["M1"]["ci"]["auc_pr"],
          V)  # fmt: skip
    m.put("vault_m1_aucroc", ci(v["models"]["M1"]["ci"]["auc_roc"]),
          v["models"]["M1"]["ci"]["auc_roc"], V)  # fmt: skip
    d = v["diffs"]["M1 - M0"]["auc_pr"]
    m.put("vault_m1_minus_m0", ci(d, signed=True), d, V)
    ts = v["text_subset"]
    m.put("vault_text_n", integer(ts["n"]), ts["n"], V)
    n_fail = round(ts["n"] * ts["failure_rate"])
    m.put("vault_text_fail_n", integer(n_fail), n_fail, V)
    d = ts["M1-texte - M1"]["auc_pr"]
    m.put("vault_m1t_minus_m1", ci(d, signed=True), d, V)
    m.put("vault_m1t_aucroc", ci(ts["M1-texte"]["ci"]["auc_roc"]), ts["M1-texte"]["ci"]["auc_roc"],
          V)  # fmt: skip
    m.put("vault_m1t_aucpr", ci(ts["M1-texte"]["ci"]["auc_pr"]), ts["M1-texte"]["ci"]["auc_pr"], V)

    # --- comité (bloc C) ---
    com = _json(COM)
    m.put("com_judged", f"{com['n_judged']} / {com['n_sample']}", com["n_judged"], COM)
    m.put("com_errors", integer(com["n_projects_with_errors"]), com["n_projects_with_errors"], COM)
    m.put("com_seconds", f"{com['median_seconds']:.0f} s", com["median_seconds"], COM)
    m.put("com_hours", num(com["elapsed_hours"], 1) + " h", com["elapsed_hours"], COM)
    q = com["quotes_found"] / com["quotes_total"]
    m.put("com_quotes", f"{integer(com['quotes_found'])} / {integer(com['quotes_total'])}",
          (com["quotes_found"], com["quotes_total"]), COM)  # fmt: skip
    m.put("com_quotes_pct", pct(q), q, COM)
    rec = com["recognition"]
    m.put("rec_recognized", f"{rec['recognized']} / {rec['n']}", rec["recognized"], COM)
    m.put("rec_rating", f"{rec['ieg_rating_proposed']} / {rec['n']}", rec["ieg_rating_proposed"],
          COM)  # fmt: skip
    m.put("rec_n", integer(rec["n"]), rec["n"], COM)
    stds = com["score_std"]
    m.put("score_std_range", f"{num(min(stds.values()), 2)} à {num(max(stds.values()), 2)}", stds,
          COM)  # fmt: skip
    m.put("share_ge9_max", pct(max(com["share_ge9"].values())), com["share_ge9"], COM)
    m.put("score_corr", num(com["mean_corr_between_criteria"], 2),
          com["mean_corr_between_criteria"], COM)  # fmt: skip
    cr = com["critic"]
    m.put("critic_weak", pct(cr["weak"] / cr["total"], 0), cr["weak"] / cr["total"], COM)
    m.put("critic_unsupported", pct(cr["unsupported"] / cr["total"]),
          cr["unsupported"] / cr["total"], COM)  # fmt: skip
    m.put("args_rewritten", pct(com["share_args_rewritten"], 0), com["share_args_rewritten"], COM)
    auc = com["valid_auc_roc_by_criterion"]
    m.put("judge_valid_auc_range", f"{num(min(auc.values()), 2)} à {num(max(auc.values()), 2)}",
          auc, COM)  # fmt: skip
    m.raw["score_mean"] = com["score_mean"]

    bc = _json(BC)
    m.put("c_n_test", integer(bc["n_test"]), bc["n_test"], BC)
    m.put("c_base", num(bc["test_failure_rate"]), bc["test_failure_rate"], BC)
    for k in ("texte", "m3", "m4"):
        m.put(f"c_{k}_aucpr", ci(bc["models"][k]["ci"]["auc_pr"]), bc["models"][k]["ci"]["auc_pr"],
              BC)  # fmt: skip
    m.put("m3_aucroc", ci(bc["models"]["m3"]["ci"]["auc_roc"]), bc["models"]["m3"]["ci"]["auc_roc"],
          BC)  # fmt: skip
    for k, key in (("m3 - m0", "m3_minus_m0"), ("m4 - texte", "m4_minus_text")):
        d = bc["diffs"][k]["auc_pr"]
        m.put(key, ci(d, signed=True), d, BC)
    m.raw["m3_weights"] = bc["weights"]["m3"]
    bcf = _json(BCF)
    m.put("c_n_valid", integer(bcf["n_valid"]), bcf["n_valid"], BCF)

    m7 = _json(M7)
    d = m7["diff_m7_texte"]["auc_pr"]
    m.put("m7_minus_text", ci(d, signed=True), d, M7)
    m7c = m7["models"]["m7"]["ci"]["auc_pr"]
    m.put("m7_aucpr", ci(m7c), m7c, M7)
    m7f = _json(M7F)
    m.put("m7_cv", num(m7f["cv_auc_pr"]), m7f["cv_auc_pr"], M7F)
    m.put("m7_cv_text", num(m7f["cv_auc_pr_text_alone"]), m7f["cv_auc_pr_text_alone"], M7F)

    # --- frontière α ---
    for tag, path in (("t", FT), ("v", FV)):
        f = _json(path)
        m.put(f"fr{tag}_pool", integer(f["n_pool"]), f["n_pool"], path)
        for row in f["frontier"]:
            a = str(row["alpha"]).replace(".", "_")
            m.put(f"fr{tag}_{a}", pct(row["count_share_satisfactory"]), row, path)
            m.put(f"fr{tag}_{a}_impact", num(row["mean_impact_sustainability"], 2),
                  row["mean_impact_sustainability"], path)  # fmt: skip
        r = f["random"]["mean"]
        m.put(f"fr{tag}_random", pct(r["count_share_satisfactory"]), f["random"], path)
        m.put(f"fr{tag}_random_impact", num(r["mean_impact_sustainability"], 2),
              r["mean_impact_sustainability"], path)  # fmt: skip
        first, last = f["frontier"][0], f["frontier"][-1]
        gain = last["mean_impact_sustainability"] - first["mean_impact_sustainability"]
        m.put(f"fr{tag}_impact_gain", num(gain, 2, signed=True), gain, path)
        loss = last["count_share_satisfactory"] - first["count_share_satisfactory"]
        m.put(f"fr{tag}_success_change", num(loss * 100, 1, signed=True) + " points", loss, path)

    # --- bloc D : extracteurs ---
    dp, df_ = _json(DP), _json(DF)
    m.put("d_pilot_cit", pct(dp["citation_valid_rate"]), dp["citation_valid_rate"], DP)
    m.put("d_pilot_s", f"{dp['median_seconds']:.0f} s", dp["median_seconds"], DP)
    m.put("d_pilot_n", integer(dp["n_projects"]), dp["n_projects"], DP)
    m.put("d_pilot_factors", f"{dp['n_factors_found_over_30pct']} / 19",
          dp["n_factors_found_over_30pct"], DP)  # fmt: skip
    m.put("d_full_n", integer(df_["n_projects"]), df_["n_projects"], DF)
    m.put("d_full_err", integer(df_["n_errors"]), df_["n_errors"], DF)
    m.put("d_full_cit", pct(df_["citation_valid_rate"]), df_["citation_valid_rate"], DF)
    m.put("d_full_s", f"{df_['median_seconds']:.0f} s", df_["median_seconds"], DF)
    m.put("d_full_much_worse", "non" if not df_["comparison_to_pilot"]["much_worse"] else "OUI",
          df_["comparison_to_pilot"]["much_worse"], DF)  # fmt: skip
    dm = _json(DM)
    m.put("d_mapping", f"{integer(dm['n_exact'])} / {integer(dm['n_keys'])}", dm["n_exact"], DM)
    pil = "reports/blocD_pilote.md"
    m.put("d_acc_env", _from_report(pil, "14 / 15"), (14, 15), pil)
    m.put("d_acc_risk", _from_report(pil, "13 / 13"), (13, 13), pil)

    d6 = _json(D6)
    m.put("d_n_test", integer(d6["n_test"]), d6["n_test"], D6)
    m.put("d_base", num(d6["test_failure_rate"]), d6["test_failure_rate"], D6)
    for k in ("texte", "m5", "m6"):
        m.put(f"d_{k}_aucpr", ci(d6["models"][k]["ci"]["auc_pr"]), d6["models"][k]["ci"]["auc_pr"],
              D6)  # fmt: skip
        m.put(f"d_{k}_aucroc", num(d6["models"][k]["ci"]["auc_roc"]["value"]),
              d6["models"][k]["ci"]["auc_roc"], D6)  # fmt: skip
    for k, key in (("m6 - texte", "m6_minus_text"), ("m5 - m0", "m5_minus_m0")):
        m.put(key, ci(d6["diffs"][k]["auc_pr"], signed=True), d6["diffs"][k]["auc_pr"], D6)
        m.put(key + "_roc", ci(d6["diffs"][k]["auc_roc"], signed=True), d6["diffs"][k]["auc_roc"],
              D6)  # fmt: skip
    m.raw["m5_weights"] = d6["weights"]["m5"]
    m.raw["m6_weights"] = d6["weights"]["m6"]
    bt = {r["model"]: r for r in d6["backtest"]}
    m.put("bt_m6", pct(bt["m6"]["count_share_satisfactory"]), bt["m6"], D6)
    m.put("bt_m6_p", pval(bt["m6"]["p_vs_random_count"]), bt["m6"]["p_vs_random_count"], D6)
    m.put("bt_text", pct(bt["texte"]["count_share_satisfactory"]), bt["texte"], D6)
    m.put("bt_text_p", pval(bt["texte"]["p_vs_random_count"]), bt["texte"]["p_vs_random_count"], D6)
    m.put("bt_random", pct(bt["hasard"]["count_share_satisfactory"]), bt["hasard"], D6)
    m.put("bt_n_pool", integer(bt["hasard"]["n_pool"]), bt["hasard"]["n_pool"], D6)
    d6f = _json(D6F)
    m.put("d_n_valid", integer(d6f["n_valid"]), d6f["n_valid"], D6F)
    m.put("d_cv_m5", num(d6f["models"]["m5"]["cv_auc_pr"]), d6f["models"]["m5"]["cv_auc_pr"], D6F)
    m.put("d_cv_m6", num(d6f["models"]["m6"]["cv_auc_pr"]), d6f["models"]["m6"]["cv_auc_pr"], D6F)
    m.put("d_cv_text", num(d6f["cv_auc_pr_text_alone"]), d6f["cv_auc_pr_text_alone"], D6F)
    m.put("d_C", num(d6f["models"]["m5"]["C"]), d6f["models"]["m5"]["C"], D6F)

    # écart entre les deux backtests de M1-texte (instabilité)
    m.put("bt_pool_gap", integer(by_budget[0.25]["n_pool"] - bt["hasard"]["n_pool"]),
          by_budget[0.25]["n_pool"] - bt["hasard"]["n_pool"], BB + " ; " + D6)  # fmt: skip
    return m


# ---------- données des graphiques (toutes tirées de load()) ----------


def chart_auc(m: Metrics) -> list[dict]:
    """AUC-PR au test avec IC 95 % et taux de base de chaque échantillon."""
    r = m.raw
    rows = [
        ("M1", "variables chiffrées ex ante", "test, projets avec texte", r["m1_on_text_aucpr"],
         r["base_test_text"], r["n_test_text"]),
        ("M1-texte", "TF-IDF du dossier complet", "test, projets avec texte", r["m1t_aucpr"],
         r["base_test_text"], r["n_test_text"]),
        ("M3", "5 notes du Juge", "test du comité", r["c_m3_aucpr"], r["c_base"], r["c_n_test"]),
        ("M4", "texte + 5 notes du Juge", "test du comité", r["c_m4_aucpr"], r["c_base"],
         r["c_n_test"]),
        ("M7", "texte + sorties du débat", "test du comité", r["m7_aucpr"], r["c_base"],
         r["c_n_test"]),
        ("M5", "19 facteurs extraits seuls", "test du bloc D", r["d_m5_aucpr"], r["d_base"],
         r["d_n_test"]),
        ("M6", "texte + facteurs extraits", "test du bloc D", r["d_m6_aucpr"], r["d_base"],
         r["d_n_test"]),
        ("M1 · coffre-fort", "projets évalués en 2025+", "données neuves", r["vault_m1_aucpr"],
         r["vault_m0_aucpr"], r["vault_n"]),
    ]  # fmt: skip
    return [{"short": a, "desc": d, "model": f"{a} · {d}", "sample": b, "value": c["value"],
             "lo": c["ci_low"], "hi": c["ci_high"], "base": base, "n": n}
            for a, d, b, c, base, n in rows]  # fmt: skip


def chart_diffs(m: Metrics) -> list[dict]:
    """Écarts appariés d'AUC-PR (même projets, même tirages bootstrap)."""
    r = m.raw
    rows = [
        ("M1 − M0", "les variables structurées battent-elles le hasard ?", r["m1_minus_m0"]),
        ("M1-texte − M1", "le texte apporte-t-il plus que les variables ?", r["m1t_minus_m1"]),
        ("M3 − M0", "les notes du Juge prédisent-elles ?", r["m3_minus_m0"]),
        ("M4 − texte", "les notes ajoutent-elles au texte ?", r["m4_minus_text"]),
        ("M7 − texte", "le débat ajoute-t-il au texte ?", r["m7_minus_text"]),
        ("M5 − M0", "les facteurs extraits prédisent-ils ?", r["m5_minus_m0"]),
        ("M6 − texte", "les facteurs ajoutent-ils au texte ?", r["m6_minus_text"]),
        ("Coffre : M1 − M0", "M1 tient-il sur des données neuves ?", r["vault_m1_minus_m0"]),
    ]
    return [{"label": a, "question": q, "value": d["value"], "lo": d["ci_low"], "hi": d["ci_high"]}
            for a, q, d in rows]  # fmt: skip


def chart_backtest(m: Metrics) -> dict:
    """Part de projets réussis des portefeuilles (25 % du budget) contre les tirages au hasard."""
    r = m.raw
    pf = r["pf25"]
    band = r["pf25_random_band"]
    return {
        "blocB": {"ours": pf["count_share_satisfactory"], "ci": pf["count_share_ci"],
                  "random_mean": pf["random_count"], "random_lo": band["p2_5"],
                  "random_hi": band["p97_5"], "m0": r["pf25_m0"]["count_share_satisfactory"],
                  "p": pf["p_count_vs_random"], "n_pool": pf["n_pool"]},
        "blocD": {"m6": r["bt_m6"]["count_share_satisfactory"],
                  "m6_ci": r["bt_m6"]["count_share_ci"],
                  "text": r["bt_text"]["count_share_satisfactory"],
                  "text_ci": r["bt_text"]["count_share_ci"],
                  "random_mean": r["bt_random"]["count_share_satisfactory"],
                  "n_pool": r["bt_random"]["n_pool"]},
    }  # fmt: skip


def chart_frontier(m: Metrics) -> dict:
    out = {}
    for tag, name in (("v", "validation"), ("t", "test")):
        rows = [m.raw[k] for k in m.raw if k.startswith(f"fr{tag}_") and k.count("_") == 2
                and isinstance(m.raw[k], dict) and "alpha" in m.raw[k]]  # fmt: skip
        rnd = m.raw[f"fr{tag}_random"]["mean"]
        out[name] = {
            "points": [{"alpha": x["alpha"], "success": x["count_share_satisfactory"],
                        "impact": x["mean_impact_sustainability"],
                        "ci": x["count_share_ci"]}
                       for x in sorted(rows, key=lambda x: -x["alpha"])],
            "random": {"success": rnd["count_share_satisfactory"],
                       "impact": rnd["mean_impact_sustainability"]},
        }  # fmt: skip
    return out


FACTOR_FR = {
    "components": "Nombre de composantes",
    "implementing_agencies": "Nombre d'agences d'exécution",
    "cofinanciers": "Nombre de cofinanceurs",
    "follow_on_phase": "Deuxième phase / suite d'un projet",
    "reform_to_adopt": "Réforme encore à adopter",
    "env_A": "Catégorie environnementale A",
    "env_B": "Catégorie environnementale B",
    "env_C": "Catégorie environnementale C",
    "env_FI": "Catégorie FI (intermédiaire financier)",
    "overall_risk_rating": "Note de risque globale",
    "fiduciary_risk": "Risque fiduciaire",
    "baseline_values": "Valeurs de référence des indicateurs",
    "lessons_problems": "Leçons : projet antérieur en difficulté",
    "fragility_conflict": "Fragilité / conflit",
    "weak_capacity": "Capacité faible de l'agence",
    "counterpart_funding": "Fonds de contrepartie requis",
    "resettlement": "Réinstallation de populations",
    "decentralized_implementation": "Exécution décentralisée",
    "pilot_innovative": "Projet pilote / innovant",
    "political_risk": "Risque politique cité",
    "economic_return_reported": "Rentabilité économique chiffrée",
}


def chart_factor_weights(m: Metrics) -> list[dict]:
    """Poids de M5 (variables standardisées) ; on écarte les indicateurs « non trouvé »."""
    w = m.raw["m5_weights"]
    rows = [{"factor": k, "label": FACTOR_FR[k], "weight": v} for k, v in w.items()
            if not k.endswith("_missing") and k in FACTOR_FR]  # fmt: skip
    return sorted(rows, key=lambda r: r["weight"], reverse=True)
