# InvestAgents

Comité d'agents IA pour l'évaluation multicritère et la sélection de projets d'investissement.
Les agents lisent les dossiers d'évaluation **ex ante** de projets financés par la Banque mondiale ;
la vérité terrain est la note de résultat de l'IEG. Spécification :
`docs/Cahier_des_charges_v3_InvestAgents.pdf`. Projet d'école, budget LLM : 0 € (modèle local).

**État : système de bout en bout ; coffre-fort évalué (M1 au niveau du hasard) ; bloc D (agents extracteurs) évalué.** Données → ML (M0, M1, M1-texte) → calibration →
sélection sous budget → comité d'agents complet (600 projets notés sur GPU Kaggle gratuit) →
M3 / M4 → démo Streamlit. **Résultats et limites : [`reports/rapport_final.md`](reports/rapport_final.md).**

En bref (test 2012–2017, évalué une fois après gel sur la validation) :
- M1-texte : AUC-PR 0,297 [0,245 ; 0,369] contre 0,216 pour le taux de base ; il bat M1 de +0,039
  [0,003 ; 0,082].
- Portefeuille sous budget (25 %) : 85,3 % de projets satisfaisants contre 77,2 % au hasard (p = 0,001).
- Comité d'agents : 99,8 % des citations vérifiées et 0/50 projet reconnu, mais des notes **non
  prédictives** (M3 au niveau du hasard ; M4 − texte = −0,031 [−0,088 ; 0,029]). Résultat négatif.
- Agents extracteurs (bloc D, 19 facteurs factuels cités) : seuls, signal faible mais réel (M5 − M0 =
  +0,027 [0,003 ; 0,070] en AUC-PR) ; en plus du texte, rien (M6 − M1-texte = +0,003 [−0,039 ; 0,043]).
- Coffre-fort (données neuves, évaluées une fois) : M1 au niveau du hasard (AUC-ROC 0,509) ; M1-texte
  non concluant (78 projets seulement, refus 403 du serveur de documents).

## Lancer la démo

```bash
uv sync
uv run streamlit run src/investagents/app/streamlit_app.py
```

La démo lit des résultats déjà calculés (`experiments/`) ; elle n'appelle aucun LLM.

## Tout recalculer

```bash
uv run python -m investagents.data.ieg                  # notes IEG
uv run python -m investagents.data.projects_api         # dates d'approbation
uv run python -m investagents.data.amounts              # montants à l'approbation
uv run python -m investagents.data.country_indicators   # WDI / CPIA / WGI
uv run python -m investagents.data.corpus               # textes des dossiers (plusieurs heures)
uv run python -m investagents.ml.run_phase2 --stage select    # réglages, validation seulement
uv run python -m investagents.ml.run_phase2 --stage evaluate  # test, une seule fois
uv run python -m investagents.selection.run_backtest    # calibration + backtest
# comité (nécessite Ollama et `ollama pull qwen2.5:3b`) :
uv run python -m investagents.agents.run_pilot --n 2
uv run pytest && uv run ruff check .
```

## Comité d'agents sur Kaggle (GPU gratuit)

Voir [`kaggle/blocC/README.md`](kaggle/blocC/README.md) : préparation du jeu
(`uv run python -m investagents.committee_run.build_dataset`), notebook généré par
`committee_run/make_kernel.py`, récupération des résultats, puis
`uv run python -m investagents.committee_run.analyze` et
`uv run python -m investagents.ml.run_blocC_models --stage select|evaluate`.

## Rapports

| Rapport | Contenu |
| --- | --- |
| `reports/rapport_final.md` | Synthèse, tableau des métriques finales, limites, attributions |
| `reports/faisabilite.md` | Phase 1 : 8 260 projets exploitables, décision GO |
| `reports/phase2_resultats.md` | M0 / M1 / M1-texte (test consulté pendant la phase 2) |
| `reports/objectif_selection.md` | Choix de l'objectif de sélection sur la validation |
| `reports/blocB_resultats.md` | Variantes de texte, masquage des pays, backtest, contrôle M0 / M1 |
| `reports/blocC_comite.md` | Comité : fonctionnement, citations, dispersion, reconnaissance |
| `reports/blocC_m3_m4.md` | M3 / M4 : apport des notes du comité (négatif) |
| `reports/blocC_m7.md` | M7 : sorties intermédiaires du débat (pas de preuve d'apport) |
| `reports/frontiere_alpha.md` | Frontière α : probabilité de réussite (ML) + notes du comité |
| `reports/blocD_pilote.md`, `reports/blocD_passage_complet.md` | Agents extracteurs : pilote, contrôle du passage complet |
| `reports/blocD_m6.md` | M5 / M6 : apport des facteurs extraits, backtest |
| `reports/coffre_fort.md` | Évaluation finale unique sur le coffre-fort (amendement documenté) |

## Limites principales

Voir `reports/rapport_final.md`. Masquage incomplet (sigles des projets, une grande ville par
document), un seul passage du comité 7B, test 2012–2017 déjà consulté en phase 2 (le chiffre neuf
viendra du coffre-fort), indicateurs pays révisés.

## Données et licences

- IEG World Bank Project Performance Ratings ; relevés IDA/IBRD ; WGI : © Banque mondiale, CC BY 4.0.
- World Bank Projects API, Documents & Reports API, Indicators API (WDI, CPIA).
- Prompts et structure du comité adaptés de DIALECTIC (github.com/pantageepapa/DIALECTIC),
  licence MIT : `src/investagents/debate/LICENSE_DIALECTIC.txt`.
- GeoNames (geonamescache) : CC BY 4.0.
- Modèles : Qwen2.5-7B-Instruct (comité, Kaggle) et Qwen2.5-3B-Instruct (développement local), via Ollama.
- Code écrit avec l'aide de Claude Code (Anthropic), relu et validé par l'autrice.
