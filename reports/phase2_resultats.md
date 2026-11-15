# Phase 2 : résultats (fichier généré automatiquement)

Généré par `uv run python -m investagents.ml.run_phase2 --stage evaluate`, après gel des
modèles sur la validation. Coffre-fort exclu. Classe positive = projet NON satisfaisant.
IC 95 % par bootstrap (2000 tirages).

| Split | Exercices | Projets | Taux d'échec |
| --- | --- | ---: | ---: |
| train | ≤ 2007 | 9004 | 28.4% |
| valid | 2008–2011 | 1243 | 22.9% |
| test | 2012–2017 | 1294 | 21.7% |

## Modèles figés (choisis sur la validation)

- M1 : variables `base+wdi`, {'learning_rate': 0.1, 'max_iter': 100, 'max_depth': 1, 'min_samples_leaf': 100, 'l2_regularization': 1.0, 'max_features': 0.5}, AUC-PR validation 0.307
- M1-texte : {'C': 1.0}, AUC-PR validation 0.355

## Au-dessus du hasard ?

- Test : 281 échecs sur 1294 -> taux de base 0.217
- Permutation M1_auc_pr : observé 0.247, hasard moyen 0.220, 95e centile 0.239, p = 0.0155
- Permutation M1_auc_roc : observé 0.567, hasard moyen 0.500, 95e centile 0.531, p = 0.0005
- Permutation M1_text_auc_pr : observé 0.332, hasard moyen 0.223, 95e centile 0.251, p = 0.0005

## Jeu de test complet (n = 1294)

| Modèle | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M0 | 0.217 [0.195 ; 0.240] | 0.500 [0.500 ; 0.500] | 0.174 [0.164 ; 0.184] |
| M1 | 0.247 [0.216 ; 0.286] | 0.567 [0.529 ; 0.603] | 0.170 [0.159 ; 0.182] |

| Modèle | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M1 - M0 | 0.030 [0.009 ; 0.060] | 0.067 [0.029 ; 0.103] | -0.003 [-0.007 ; 0.000] |

## Sous-ensemble du test avec texte (n = 755)

| Modèle | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M0 | 0.216 [0.187 ; 0.245] | 0.500 [0.500 ; 0.500] | 0.173 [0.160 ; 0.186] |
| M1 | 0.278 [0.229 ; 0.340] | 0.594 [0.543 ; 0.640] | 0.167 [0.152 ; 0.183] |
| M1_text | 0.332 [0.272 ; 0.409] | 0.631 [0.581 ; 0.680] | 0.162 [0.146 ; 0.179] |

| Modèle | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M1_text - M1 | 0.054 [0.006 ; 0.112] | 0.038 [-0.002 ; 0.074] | -0.005 [-0.008 ; -0.001] |
| M1_text - M0 | 0.116 [0.069 ; 0.185] | 0.131 [0.081 ; 0.179] | -0.011 [-0.016 ; -0.005] |

## Diagnostics de fuite et de surapprentissage

- M1 entraîné sur train : AUC-PR train 0.403, valid 0.307
- M1-texte entraîné sur train : AUC-PR train 0.870, valid 0.355

Importance par permutation (M1, test, baisse d'AUC-PR) :

| variable                |    mean |    std |
|:------------------------|--------:|-------:|
| wb_region               |  0.0262 | 0.0086 |
| log_population          |  0.0012 | 0.0035 |
| practice_group          |  0.0000 | 0.0000 |
| agreement_type          |  0.0000 | 0.0000 |
| lending_instrument_type |  0.0000 | 0.0000 |
| inflation               |  0.0000 | 0.0000 |
| approval_fy             |  0.0000 | 0.0000 |
| amount_missing          |  0.0000 | 0.0000 |
| oda_gni                 | -0.0004 | 0.0013 |
| log_gdp_pc              | -0.0014 | 0.0034 |
| log_amount_at_approval  | -0.0026 | 0.0019 |
| global_practice         | -0.0027 | 0.0073 |
| gdp_growth              | -0.0036 | 0.0039 |

Mots qui poussent vers l'échec : cameroon, subloans, zaire, subloan, contained, frameworks, algeria, des, concession, africa, ppp, losses, unions, livestock, sofide, au, forestry, smallholder, managerial, sewage, workshop, fmr, instruments, african, metering, mining, apparently, integrity, chad, dam

Mots qui poussent vers le succès : china, emergency, thailand, rs, armenia, vietnam, korea, tax, contracting, traffic, thousands, eu, esmf, provinces, nam, chile, geographic, temporary, pavement, substation, feeder, beneficiary, cadastre, rapid, mountainous, pages, measurement, peace, spot, gor

Mots « suspects » (vocabulaire de fin de projet) présents dans le modèle :

| mot            |   coef |
|:---------------|-------:|
| closing        | -0.299 |
| cancelled      | -0.185 |
| cancellation   | -0.111 |
| restructured   | -0.071 |
| rated          | -0.065 |
| outcome        | -0.047 |
| unsatisfactory |  0.092 |
| rating         |  0.177 |
| ppar           |  0.199 |
| restructuring  |  0.229 |
| icr            |  0.239 |
| extension      |  0.327 |
