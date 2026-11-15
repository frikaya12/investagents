# Phase 2 expliquée simplement : chaque fichier, ce qu'il fait, pourquoi

*Les chiffres de résultat sont dans `reports/phase2_resultats.md` (généré par le code) et résumés
dans la section « Résultats » à la fin de ce document.*

## 1. Le montant à l'approbation — `src/investagents/data/amounts.py`

**Problème.** Les montants de la Projects API pourraient inclure des financements votés après
l'approbation : ce serait une fuite (un projet qui marche reçoit souvent un financement
additionnel).

**Solution.** Les relevés officiels des prêts de la Banque mondiale (FinancesOne, CC BY 4.0) :
IDA (DS00001, 11 509 crédits) et IBRD (DS00047, 9 545 prêts). Chaque ligne est un prêt avec son
**montant initial** (`original_principal_amount`) et sa date d'approbation. Pour chaque projet, on
additionne les prêts approuvés à la **première** date d'approbation ; ceux approuvés plus tard
(1 691 projets en ont) sont exclus.

**Vérification faite.** Sur 10 484 projets notés comparables, `totalamt` de la Projects API est égal
à ce montant initial (±1 %) dans 97,3 % des cas : `totalamt` est donc *en pratique* le montant à
l'approbation, mais ce n'est documenté nulle part, d'où le choix des relevés officiels.
Couverture : 11 566 projets notés sur 12 511 (100 % des IBRD et IDA ; les projets GEF et fonds
fiduciaires n'ont pas de montant → variable `amount_missing`).

## 2. Les variables ex ante — `src/investagents/ml/features.py`

| Variable | Type | Source | Pourquoi elle est ex ante |
| --- | --- | --- | --- |
| `wb_region` | catégorie | IEG | région du projet, connue dès la conception |
| `practice_group`, `global_practice` | catégorie | IEG | secteur du projet |
| `lending_instrument_type` | catégorie | IEG | IPF (projet) ou DPF (appui budgétaire), choisi à la conception |
| `agreement_type` | catégorie | IEG | IBRD, IDA, GEF… : guichet de financement |
| `country` | catégorie | IEG | pays emprunteur |
| `approval_fy` | nombre | IEG | exercice d'approbation |
| `log_amount_at_approval` | nombre | relevés IDA/IBRD | log du montant initial (les montants vont de quelques millions à des milliards) |
| `amount_missing` | 0/1 | calculée | 1 si pas de montant (GEF, fonds fiduciaires) |

**Interdites** (liste `FORBIDDEN_EXACT` + préfixe `evaluation_`) : `quality_at_entry`,
`bank_performance`, `quality_of_supervision`, `me_quality`, `final_closing_fy`, `evaluation_*`
(jugées après le projet), la cible, et **par prudence** `country_fcs_status`,
`country_fcs_lending_group`, `country_lending_group` : on ne sait pas à quelle date l'IEG mesure le
statut du pays. Le statut varie dans le temps pour un même pays (51 pays sur 187), mais rien dans
les données ne dit s'il est pris à l'approbation ou à l'évaluation ; l'Éthiopie, par exemple, est
« FCS » pour un projet approuvé en 2019 et « non-FCS » avant. Si c'est le statut à l'évaluation,
ce serait une fuite. Dans le doute, on exclut (à vérifier dans la documentation IEG).

**La garde.** `build_features` appelle `assert_no_forbidden` sur ses colonnes de sortie : si
quelqu'un ajoute une colonne interdite, le programme s'arrête. Le test
`test_adding_a_forbidden_column_to_features_fails` simule exactement cette erreur et vérifie que
ça échoue.

**La cible.** `failure` = 1 si le projet est non satisfaisant (MU/U/HU). On prédit l'échec, la
classe minoritaire (~25 %), parce que c'est ce qu'un bailleur veut repérer, et parce que l'AUC-PR
n'a de sens que sur la classe rare (sur la classe majoritaire, un modèle au hasard ferait déjà 0,75).

## 3. Le coffre-fort et le split — `src/investagents/ml/split.py`

**Coffre-fort.** Les 700 projets évalués en exercice ≥ 2025 sont retirés dès le chargement
(`load_modeling_table`) : aucune fonction de modélisation ne peut les voir. Leurs textes ne sont pas
téléchargés. Un test le vérifie.

**Split temporel** (sur les 11 811 projets restants) :

| Jeu | Exercices d'approbation | Rôle |
| --- | --- | --- |
| Entraînement | ≤ 2007 | apprendre |
| Validation | 2008–2011 | choisir les hyperparamètres |
| Test | 2012–2017 | mesure finale, utilisée une seule fois |
| Écarté | ≥ 2018 | cohorte biaisée |

**Pourquoi ces années.**
- *Test 2012–2017* : la période la plus récente où la cohorte est **complète et représentative**
  (~250 projets notés par an, ~25 % de DPF). C'est la situation réelle : prédire des projets
  récents à partir du passé.
- *≥ 2018 écarté* : beaucoup de projets approuvés après 2018 sont encore en cours, et ceux déjà
  évalués sont surtout des appuis budgétaires courts (57 % de DPF en 2018–19, 76 % en 2020–21,
  évalués en 3 ans médians au lieu de 7). Les garder biaiserait le test. (Les évaluations ≥ 2025,
  le gros de ces années, sont de toute façon dans le coffre-fort.)
- *Validation 2008–2011* : juste avant le test, pour régler les modèles dans des conditions
  proches ; ~1 200 projets, assez pour un réglage stable.
- *Entraînement ≤ 2007* : tout le reste (~9 000 projets).
Après réglage, le modèle final est réentraîné sur entraînement + validation (≤ 2011).

**Limite assumée.** Un projet approuvé en 2007 a été évalué vers 2015 : en 2012, on n'aurait pas
encore connu son issue. Le split par date d'approbation est donc un peu optimiste par rapport à
la réalité. Ça vaut pour tous les modèles de la même façon, donc la comparaison reste juste.

## 4. Le corpus de textes — `src/investagents/data/corpus.py`

Pour chaque projet hors coffre-fort, on prend **un** dossier d'évaluation : tier 1, gardé par
`leakage.py` (anglais, daté strictement avant l'approbation), le plus récent (la version finale
soumise au Conseil). Texte téléchargé et stocké compressé. Le serveur bloque (erreur 403) si on va
trop vite : le téléchargement attend 15 s, 30 s, 60 s... et réessaie, avec 2 connexions au plus.

## 5. Les métriques — `src/investagents/ml/metrics.py`

- **AUC-PR** (métrique principale) : aire sous la courbe précision-rappel sur la classe échec.
  Hasard = taux d'échec du test (~0,22).
- **AUC-ROC** : probabilité qu'un projet en échec ait un score plus haut qu'un projet réussi.
  Hasard = 0,5.
- **Brier** : erreur quadratique moyenne des probabilités ; mesure aussi la calibration.
- **IC 95 % par bootstrap** : on retire 2 000 fois au hasard (avec remise) les projets du test et
  on recalcule ; l'intervalle contient 95 % des valeurs.
- **Écart apparié** : pour comparer deux modèles, on calcule leur différence sur les *mêmes*
  tirages. Si l'IC de l'écart exclut 0, la différence est significative.

## 6. Les modèles — `src/investagents/ml/models.py`

- **M0** : prédit le même taux d'échec pour tous les projets (celui de l'entraînement). AUC-ROC
  = 0,5 par construction : c'est le plancher.
- **M1** : gradient boosting d'arbres (`HistGradientBoostingClassifier`), qui gère nativement les
  catégories et les valeurs manquantes. Réglage sur la validation par AUC-PR (162 combinaisons).
- **M1-texte** : TF-IDF (poids des mots, mots outils anglais retirés, lettres seulement donc pas de
  chiffres, d'années ni d'identifiants, 150 000 premiers caractères, 50 000 mots au plus) + régression
  logistique. Le vocabulaire et les poids IDF sont appris **sur l'entraînement seulement**. `C` (force
  de régularisation) réglé sur la validation.

**Comparaison équitable.** M1-texte n'existe que pour les projets qui ont un texte. On compare
donc M1-texte et M1 sur le **même sous-ensemble** du test.

## 7. L'expérience — `src/investagents/ml/run_phase2.py`

Enchaîne tout : chargement (coffre-fort exclu), variables, réglage, réentraînement, prédictions,
IC, importance par permutation (combien l'AUC-PR baisse quand on mélange une variable), mots les plus
influents, et une liste de mots « suspects » (vocabulaire de fin de projet : *completion*,
*restructuring*, *rating*…) pour détecter une fuite par le texte. Écrit `experiments/phase2/` et
`reports/phase2_resultats.md`.

## Tests ajoutés

`test_amounts.py`, `test_features_split.py` (colonnes interdites, coffre-fort, split),
`test_corpus.py`, `test_metrics_models.py` (métriques, bootstrap, et modèles sur données
synthétiques à règle cachée : M1 doit retrouver « échec si région A et petit montant »).

## Pour tout relancer

```bash
uv run python -m investagents.data.amounts
uv run python -m investagents.data.corpus      # long : plusieurs heures
uv run python -m investagents.ml.run_phase2
```
