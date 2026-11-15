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
| `approval_fy` | nombre | IEG | exercice d'approbation |
| `log_amount_at_approval` | nombre | relevés IDA/IBRD | log du montant initial (les montants vont de quelques millions à des milliards) |
| `amount_missing` | 0/1 | calculée | 1 si pas de montant (GEF, fonds fiduciaires) |
| `log_gdp_pc`, `gdp_growth`, `inflation`, `log_population`, `oda_gni` | nombres | WDI | situation du pays, valeur d'une année ≤ approbation − 2 (groupe « wdi ») |
| `cpia` | nombre | WDI (CPIA) | qualité des politiques du pays, 1 à 6 (groupe « cpia », non retenu) |
| `wgi_cc` … `wgi_va` | nombres | WGI | 6 indicateurs de gouvernance (groupe « wgi », non retenu) |

**Variable `country` retirée** (décision d'Aya) : à performance égale sur le test (AUC-PR 0,268
sans pays, 0,270 avec), on garde le modèle le plus simple ; le pays (~180 modalités) servait
surtout à mémoriser l'entraînement. Transparence : sur la validation, le modèle avec pays faisait
mieux (0,308 contre 0,280) ; cette décision a donc été prise après avoir vu le test.

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
Résultat : 7 518 projets visés, **7 434 textes exploitables** (≥ 10 000 caractères), 4 échecs.

## 4 bis. Les indicateurs pays-année — `src/investagents/data/country_indicators.py`

- **WDI** (API Indicators, source 2) : PIB/hab., croissance, inflation, population, aide reçue
  en % du RNB, et le **CPIA** (indice d'allocation IDA, publié depuis 2005).
- **WGI** : 6 indicateurs de gouvernance (fichier complet du DataBank, 1996–2025 ; l'API WGI
  était indisponible ce jour-là).
- **Règle ex ante** : pour un projet approuvé en exercice N, dernière valeur d'une année ≤ N − 2
  (une valeur de l'année N − 1 est souvent publiée après l'approbation), au plus 4 ans d'âge.
  Testé : aucune valeur future, ancienneté maximale respectée.
- **Pays du projet** : code pays des relevés de prêts IDA/IBRD (les codes régionaux comme « 3E »
  ou « 6R » sont écartés).
- **Couverture** : WDI ~90 % à l'entraînement ; WGI 26 % (débute en 1996) ; **CPIA 2 %** à
  l'entraînement (débute en 2005) : le modèle ne peut presque rien apprendre du CPIA.
- **Limite** : ce sont les valeurs actuelles des bases (révisées depuis), pas les valeurs connues
  à l'époque.

**Choix sur la validation seule** (AUC-PR de validation, chaque ensemble réglé avec 324
combinaisons) : base 0,299 ; **base + WDI 0,307 (retenu)** ; base + CPIA 0,302 ; base + WGI
0,304 ; tout 0,304. Les écarts sont faibles (< 0,01) et probablement du bruit.

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
  catégories et les valeurs manquantes. Réglage sur la validation par AUC-PR, grille orientée
  vers une forte régularisation (324 combinaisons : profondeur 1 à 3, feuilles de 100 à 600
  projets, learning rate 0,01 à 0,1, pénalité L2, moitié ou totalité des variables par arbre).
  Réglage retenu : arbres à un seul niveau (`max_depth = 1`), donc un modèle additif sans
  interactions. Écart AUC-PR entraînement / validation : 0,40 / 0,31.
- **M1-texte** : TF-IDF (poids des mots, mots outils anglais retirés, lettres seulement donc pas de
  chiffres, d'années ni d'identifiants, 150 000 premiers caractères, 50 000 mots au plus) + régression
  logistique. Le vocabulaire et les poids IDF sont appris **sur l'entraînement seulement**. `C` (force
  de régularisation) réglé sur la validation.

**Comparaison équitable.** M1-texte n'existe que pour les projets qui ont un texte. On compare
donc M1-texte et M1 sur le **même sous-ensemble** du test.

## 7. L'expérience — `src/investagents/ml/run_phase2.py`

Deux étapes séparées, pour garantir que le test ne sert à aucun choix :
1. `--stage select` : validation seulement. Choisit l'ensemble de variables et les
   hyperparamètres de M1, et le `C` de M1-texte ; écrit `experiments/phase2/frozen.json`.
2. `--stage evaluate` : relit `frozen.json`, réentraîne sur ≤ 2011, calcule le test **une fois**.
   Métriques, IC, écarts appariés, **test de permutation** (on mélange 2 000 fois les étiquettes
   pour voir ce que fait le hasard), importance par permutation, mots influents et liste de mots
   « suspects » (vocabulaire de fin de projet) pour détecter une fuite par le texte.

Les chiffres définitifs de tous les modèles seront calculés une seule fois sur le coffre-fort, à
la fin du projet.

## Tests ajoutés

`test_amounts.py`, `test_features_split.py` (colonnes interdites, coffre-fort, split, groupes
pays), `test_country_indicators.py` (décalage N − 2, pas de valeur future, codes régionaux),
`test_corpus.py` (dont l'attente après un 403), `test_metrics_models.py` (métriques, bootstrap,
test de permutation, et modèles sur données synthétiques à règle cachée : M1 doit retrouver
« échec si région A et petit montant »). Total : 101 tests hors réseau.

## Pour tout relancer

```bash
uv run python -m investagents.data.amounts
uv run python -m investagents.data.country_indicators
uv run python -m investagents.data.corpus                      # long : plusieurs heures
uv run python -m investagents.ml.run_phase2 --stage select     # validation seulement
uv run python -m investagents.ml.run_phase2 --stage evaluate   # test, une seule fois
```

## 8. Résultats (test 2012–2017, calculé une seule fois après gel)

Classe positive : **échec** (MU/U/HU). Taux de base du test : 281 / 1 294 = **0,217**.
Chiffres exacts et IC : `reports/phase2_resultats.md`.

| Jeu | Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier |
| --- | --- | --- | --- | --- |
| Test complet (1 294) | M0 | 0,217 [0,195 ; 0,240] | 0,500 | 0,174 |
| | M1 (base + WDI) | 0,247 [0,216 ; 0,286] | 0,567 [0,529 ; 0,603] | 0,170 |
| Test avec texte (755) | M0 | 0,216 [0,187 ; 0,245] | 0,500 | 0,173 |
| | M1 | 0,278 [0,229 ; 0,340] | 0,594 [0,543 ; 0,640] | 0,167 |
| | **M1-texte** | **0,332 [0,272 ; 0,409]** | **0,631 [0,581 ; 0,680]** | 0,162 |

Écarts appariés : M1 − M0 = +0,030 [0,009 ; 0,060] en AUC-PR (test complet) ;
M1-texte − M1 = +0,054 [0,006 ; 0,112] en AUC-PR et +0,038 [−0,002 ; 0,074] en AUC-ROC (même
sous-ensemble). Tests de permutation : M1 p = 0,016 (AUC-PR) ; M1-texte p = 0,0005.

**Lecture.**
- Les variables structurées ex ante ne portent qu'un **signal faible** mais réel (M1 bat le hasard,
  IC de l'écart > 0). Les indicateurs pays (WDI, CPIA, WGI) n'y changent presque rien.
- Le **texte du dossier d'évaluation apporte plus** : M1-texte bat M1 en AUC-PR (IC de l'écart
  > 0), mais l'écart en AUC-ROC n'est pas significatif (IC touche 0). Gain réel mais modeste et
  incertain.
- Les mots les plus influents de M1-texte sont souvent des **noms de pays** (Cameroon, Zaire,
  Chad / China, Korea, Vietnam) : une partie du gain du texte vient sans doute de l'identité du
  pays, que l'on a retirée de M1. À garder en tête pour interpréter M4 en phase 4.
- Le sous-ensemble avec texte est un peu plus « facile » pour M1 aussi (0,278 contre 0,247 sur
  tout le test) : les projets sans texte (GEF, fonds fiduciaires, dossiers absents) sont plus
  durs à prédire.

## 9. Diagnostics de fuite

| Contrôle | Résultat | Conclusion |
| --- | --- | --- |
| Un AUC « trop beau » ? | AUC-ROC 0,57 (M1) et 0,63 (M1-texte) | Ordres de grandeur plausibles, aucun signe d'alerte |
| Colonnes post-projet dans les variables | Garde `assert_no_forbidden` + test | Impossible par construction |
| Mots de fin de projet dans M1-texte (*icr*, *ppar*, *restructuring*, *extension*…) | Présents, coefficients ±0,3 | Contexte vérifié sur 800 dossiers : liste des sigles, rapport « prévu », leçons de projets passés : ex ante |
| Même modèle sans ces mots (validation) | AUC-PR 0,3548 contre 0,3551 | Aucun effet : la performance ne vient pas de ce vocabulaire |
| Écart entraînement / validation | M1 0,40 / 0,31 ; M1-texte 0,87 / 0,36 | Surapprentissage attendu en entraînement (surtout pour 50 000 mots), mais la validation, hors échantillon, reste au-dessus du hasard |
| Coffre-fort | 700 projets jamais chargés, textes non téléchargés | Intact |

## 10. Dérive du taux d'échec (à traiter en calibration)

Le taux d'échec baisse dans le temps : **28,4 %** (entraînement ≤ 2007) → **22,9 %** (validation)
→ **21,7 %** (test). Conséquence visible : M0 prédit 27,7 % pour tous les projets du test, d'où un
Brier à peine meilleur pour M1. Les probabilités de tous les modèles sont donc **trop
pessimistes** sur la période récente. À corriger lors de la calibration (phase 4/5), par exemple
en recalibrant sur la période la plus récente disponible (la validation), jamais sur le test.
