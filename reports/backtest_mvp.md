# Calibration et backtest de portefeuille (MVP, généré automatiquement)

Généré par `uv run python -m investagents.selection.run_backtest`. Modèles entraînés sur
≤ 2007, recalibrés (Platt) sur la validation 2008–2011, évalués sur le test 2012–2017.
Aucun réglage sur le test.

## Calibration (test, classe « échec »)

| Modèle | Brier avant | Brier après | ECE avant | ECE après | Prédiction moyenne avant → après | Taux d'échec réel |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| m1 | 0.1780 | 0.1713 | 0.0750 | 0.0443 | 0.292 → 0.230 | 0.217 |
| m1_text | 0.1679 | 0.1662 | 0.0447 | 0.0234 | 0.261 → 0.203 | 0.216 |

## Backtest (728 projets du test avec texte, montant et pays ; plafond 25% du budget par pays)

Part de projets **satisfaisants** dans le portefeuille, toutes cohortes réunies.

| Budget | M1-texte [IC 95 %] | M1 [IC 95 %] | Hasard (1 000 tirages) [2,5 % ; 97,5 %] | Taux de base | p (M1-texte vs hasard) |
| --- | --- | --- | --- | ---: | ---: |
| 10% | 75.8% [70.9% ; 80.2%] (n = 327) | 74.6% [69.6% ; 79.0%] (n = 327) | 76.4% [69.4% ; 82.7%] | 78.0% | 0.573 |
| 25% | 76.0% [72.0% ; 79.6%] (n = 488) | 76.0% [72.0% ; 79.6%] (n = 488) | 77.3% [72.8% ; 81.8%] | 78.0% | 0.708 |
| 50% | 77.5% [74.1% ; 80.7%] (n = 619) | 77.4% [73.9% ; 80.5%] (n = 619) | 77.7% [74.9% ; 80.4%] | 78.0% | 0.544 |
