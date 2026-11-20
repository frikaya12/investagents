# InvestAgents

Comité d'agents IA pour l'évaluation multicritère et la sélection de projets d'investissement.
Les agents lisent les dossiers d'évaluation **ex ante** de projets financés par la Banque mondiale ;
la vérité terrain est la note de résultat de l'IEG. Spécification :
`docs/Cahier_des_charges_v3_InvestAgents.pdf`. Projet d'école, budget LLM : 0 € (modèle local).

**État : première version de bout en bout (MVP).** Données → modèles ML → calibration →
sélection sous budget → comité d'agents (pilote sur 2 projets) → démo Streamlit.

## Lancer la démo

```bash
uv sync
uv run streamlit run src/investagents/app/streamlit_app.py
```

La démo lit des résultats déjà calculés (`experiments/mvp/`) ; elle n'appelle aucun LLM.

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

## Ce qui marche

| Brique | Fichiers | État |
| --- | --- | --- |
| Données IEG, Projects API, Documents API, montants à l'approbation, indicateurs pays | `src/investagents/data/` | fait, testé |
| Filtre anti-fuite (documents ex ante datés avant l'approbation) | `data/leakage.py` | fait, 100 % couvert |
| Split temporel + coffre-fort (évaluations ≥ 2025 jamais touchées) | `ml/split.py` | fait |
| M0, M1 (gradient boosting), M1-texte (TF-IDF + régression logistique), figés sur la validation | `ml/` | fait |
| Recalibration de Platt (validation), Brier et ECE | `ml/calibration.py` | fait |
| Sélection sous budget, sac à dos OR-Tools CP-SAT, ≤ 25 % du budget par pays | `selection/knapsack.py` | fait |
| Backtest 10 / 25 / 50 % du budget contre hasard, M1, taux de base | `selection/run_backtest.py` | fait |
| Comité compact LangGraph : sections clés → débat Pour/Contre (5 critères, citations) → Juge 0-10 → vérification des citations | `graph/committee.py`, `agents/`, `debate/` | pilote sur 2 projets |
| Démo Streamlit | `app/streamlit_app.py` | fait |

## Métriques actuelles

Jeu de test = projets approuvés en 2012–2017 (le coffre-fort, évalué en 2025 ou après, est réservé
au chiffre final). Classe « positive » des AUC = projet **non satisfaisant**. Rapports détaillés :
`reports/phase2_resultats.md`, `reports/backtest_mvp.md`.

### Métriques du MVP

Chaque chiffre vient d'un script du dépôt. IC 95 % : bootstrap (2 000 tirages) pour les AUC, Wilson
pour les proportions, centiles 2,5 / 97,5 % pour le hasard (1 000 portefeuilles).

| # | Métrique | Valeur [IC 95 %] | Référence | Ce que ça prouve |
| --- | --- | --- | --- | --- |
| 1 | Projets exploitables (dossier ex ante en texte, daté avant approbation) | 8 260 (recensement) | seuil 300 | La faisabilité est largement acquise |
| 2 | AUC-PR M1, test complet (n = 1 294) | 0,247 [0,216 ; 0,286] | hasard 0,217 | Les variables ex ante portent un signal faible mais réel (p = 0,016) |
| 3 | AUC-ROC M1, test complet | 0,567 [0,529 ; 0,603] | 0,5 | Idem |
| 4 | AUC-PR M1-texte, test avec texte (n = 755) | 0,332 [0,272 ; 0,409] | M1 0,278 ; hasard 0,216 | Le texte du dossier d'évaluation apporte de l'information |
| 5 | Écart M1-texte − M1 (AUC-PR, apparié) | +0,054 [0,006 ; 0,112] | 0 | Gain du texte significatif en AUC-PR (pas en AUC-ROC : +0,038 [−0,002 ; 0,074]) |
| 6 | ECE M1-texte, test, avant → après recalibration (Platt, validation) | 0,045 → 0,023 | 0 = parfait | La recalibration sur la validation corrige la dérive du taux d'échec |
| 7 | Brier M1-texte, test, avant → après | 0,168 → 0,166 | M0 0,173 | Probabilités un peu meilleures que le taux de base |
| 8 | Objectif initial Σ p, part de projets satisfaisants (budget 10 / 25 / 50 %) | 75,8 / 76,0 / 77,5 % | hasard 76,5 / 77,2 / 77,7 % | **Pas mieux que le hasard** (p = 0,58 / 0,70 / 0,55) : il favorise les petits projets |
| 9 | **Objectif figé Σ p × coût** (choisi sur la validation), part de projets satisfaisants, budget 25 % | **85,3 %** [79,3 ; 90,7] | hasard 77,2 % [72,6 ; 82,1] ; base 78,0 % | Meilleur que le hasard (p = 0,001) |
| 10 | Idem, budget 50 % ; part du budget allant à des projets satisfaisants | 83,9 % [79,1 ; 88,3] (p = 0,001) ; budget 85,7 % [79,6 ; 91,1] | hasard 77,7 % ; budget 82,1 % | Meilleur que le hasard en nombre ; en montant, gain limite (p = 0,050). À 10 % : pas significatif (p = 0,13 et 0,22) |
| 11 | Citations valides du comité, citations par numéros de phrases (2 projets, 20 citations) | **100 %** [84 ; 100] | version précédente (citations recopiées) : 25 % [11 ; 47] | Les citations existent toutes dans le document ; leur pertinence reste à juger |
| 12 | Sorties JSON valides du comité au 1er essai | 8 / 8 appels (2 versions × 2 projets × 2 appels) | — | Pydantic + schéma Ollama : format fiable |
| 13 | Temps par projet (comité, qwen2.5:3b, CPU, sans GPU) | 4,6 et 5,0 min | — | ~5 min/projet : 300 projets ≈ 25 h de calcul local |
| 14 | Stabilité des notes (3 passages × 10 projets) | non mesurée | — | À faire |
| 15 | Taux de reconnaissance des projets par le LLM | non mesuré | — | À faire avant toute conclusion sur les agents |

**Objectif de sélection.** L'objectif initial (Σ p_i sous budget) récompense surtout le **nombre** de
projets : le solveur prend les moins chers et ne bat pas le hasard. Trois objectifs ont été comparés
**sur la validation** avec un critère fixé à l'avance (part du budget allant à des projets satisfaisants) :
Σ p (79,2 %), Σ p × coût (95,1 %), Σ p × coût avec p ≥ taux de base (95,1 %). Σ p × coût a été figé, puis
évalué **une seule fois** sur le test (lignes 9 et 10). Le test est nettement en dessous de la
validation (attendu : M1-texte et sa calibration ont été réglés sur la validation). Détail :
`reports/objectif_selection.md`.

**Contrôle : d'où vient le gain ?** Même objectif figé, sur le test, avec d'autres probabilités.
Avec des probabilités constantes (M0 : seule la taille compte), le portefeuille ne bat pas le hasard
(p = 0,64 à 25 % du budget). Avec M1 (sans texte), le gain est proche de celui de M1-texte (81,8 %
contre 85,3 % de projets satisfaisants à 25 % du budget, intervalles qui se recouvrent). Le gain vient
donc des probabilités du modèle, pas de la préférence pour les gros projets, et pas spécifiquement du
texte.

## Limites (à lire avant de présenter)

- **Signal faible.** Les variables ex ante prédisent peu l'issue d'un projet (AUC-ROC 0,57 pour M1) ;
  le texte aide un peu (0,63), mais l'écart avec M1 n'est significatif qu'en AUC-PR.
- **Comité = pilote.** 2 projets seulement, modèle local de 3 milliards de paramètres sur un portable
  sans GPU : environ 5 minutes par projet. Les citations par numéros de phrases sont toutes valides,
  mais « valide » veut dire que la phrase existe, pas qu'elle soutient l'argument. Le Juge 3B semble
  peu discriminant (un projet reçoit 10/10 sur 4 critères). Les notes ne sont pas encore utilisables
  comme variables.
- **Objectif de sélection** choisi sur la validation : le gain sur le test est réel en nombre de
  projets (25 % et 50 % du budget) mais limite en montant, et absent à 10 % du budget.
- **Pas encore mesurés** : stabilité des notes (plusieurs passages), test de reconnaissance des projets
  par le LLM, M2 / M3 / M4, ablations.
- **Test consulté plusieurs fois** pendant la phase 2 (documenté) : les chiffres de test sont
  indicatifs ; le chiffre définitif viendra du coffre-fort.
- Indicateurs pays (WDI, WGI) : valeurs révisées actuelles, pas celles connues à l'époque.

## Suite prévue

1. Comité : mesurer la stabilité (3 passages × 20 projets), le taux de reconnaissance, et un contrôle
   de pertinence des phrases citées.
2. Plus de projets pour entraîner M3 / M4 (notes du Juge + variables) : GPU gratuit (Kaggle) ou
   modèle 3B local la nuit.
3. Variante M1-texte avec noms de pays masqués ; modèle combiné M1 + M1-texte réglé sur la validation.
4. Chiffres définitifs sur le coffre-fort, une seule fois.

## Données et licences

- IEG World Bank Project Performance Ratings ; relevés IDA/IBRD ; WGI : © Banque mondiale, CC BY 4.0.
- World Bank Projects API, Documents & Reports API, Indicators API (WDI, CPIA).
- Prompts et structure du comité adaptés de DIALECTIC (github.com/pantageepapa/DIALECTIC),
  licence MIT : `src/investagents/debate/LICENSE_DIALECTIC.txt`.
- Modèle local : Qwen2.5-3B-Instruct via Ollama.
