# Phase 2 : résultats bruts (fichier généré automatiquement)

Généré par `uv run python -m investagents.ml.run_phase2`. Coffre-fort exclu.
Classe positive = projet NON satisfaisant. IC 95 % par bootstrap (2000 tirages).

| Split | Exercices d'approbation | Projets | Taux d'échec |
| --- | --- | ---: | ---: |
| train | ≤ 2007 | 9004 | 28.4% |
| valid | 2008–2011 | 1243 | 22.9% |
| test | 2012–2017 | 1294 | 21.7% |

## Au-dessus du hasard ?

- Classe positive : failure = non satisfaisant (MU/U/HU)
- Test : 281 positifs sur 1294 -> taux de base 0.217 (= AUC-PR d'un classement au hasard)
- Test de permutation M1 AUC-PR : observé 0.246, hasard moyen 0.219, 95e centile du hasard 0.237, p = 0.0090
- Test de permutation M1 AUC-ROC : observé 0.559, hasard moyen 0.500, 95e centile du hasard 0.532, p = 0.0030

## Jeu de test complet (n = 1294)

| Modèle | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M0 | 0.217 [0.195 ; 0.240] | 0.500 [0.500 ; 0.500] | 0.174 [0.164 ; 0.184] |
| M1 | 0.246 [0.214 ; 0.284] | 0.559 [0.522 ; 0.598] | 0.171 [0.160 ; 0.182] |

| Écart apparié | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M1 - M0 | 0.029 [0.007 ; 0.056] | 0.059 [0.021 ; 0.096] | -0.003 [-0.006 ; 0.000] |

## Diagnostics

- M1 hyperparamètres : {'learning_rate': 0.03, 'max_iter': 200, 'max_depth': 1, 'min_samples_leaf': 100, 'l2_regularization': 1.0, 'max_features': 0.5}
- M1 (entraîné sur train) : AUC-PR train 0.386, valid 0.299

Importance par permutation (M1, test, baisse d'AUC-PR) :

| variable                |    mean |    std |
|:------------------------|--------:|-------:|
| wb_region               |  0.0318 | 0.0085 |
| agreement_type          |  0.0006 | 0.0017 |
| lending_instrument_type |  0.0004 | 0.0012 |
| practice_group          |  0.0002 | 0.0016 |
| amount_missing          |  0.0000 | 0.0000 |
| approval_fy             |  0.0000 | 0.0000 |
| log_amount_at_approval  | -0.0045 | 0.0031 |
| global_practice         | -0.0072 | 0.0076 |
