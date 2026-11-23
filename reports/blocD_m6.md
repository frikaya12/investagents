# Bloc D, M5 et M6 : lecture (écrit à la main)

**Question : les facteurs de risque extraits par les agents apportent-ils une information ?**
Les choix de codage ont été figés avant le réglage (experiments/blocD/m6_choices.json). Réglage par
validation croisée sur les 853 projets de validation ; test une seule fois sur les 729.

- **Facteurs seuls (M5) : oui, un signal faible mais réel. C'est le premier apport prédictif
  mesurable des agents dans ce projet.**
  - M5 − M0 : +0,027 [0,003 ; 0,070] en AUC-PR et +0,066 [0,017 ; 0,114] en AUC-ROC. L'AUC-ROC
    de M5 vaut 0,566, du même ordre que M1 sur les variables structurées (0,567 au test complet).
  - La validation l'annonçait aussi (0,248 contre un taux de base de 0,227).
- **En plus du texte (M6) : non.** M6 − M1-texte = +0,003 [−0,039 ; 0,043] en AUC-PR, Brier
  identique. Les facteurs ne disent rien de plus que le TF-IDF du dossier entier, ce qui est
  cohérent : ils sont lus dans le même document. La validation l'annonçait (M6 0,327 contre 0,337
  pour le texte seul), et le réglage a choisi la régularisation la plus forte de la grille
  (C = 0,003, au bord).
- **Backtest (25 % du budget, objectif figé).**
  - M6 donne 86,1 % de projets réussis [80,6 ; 91,1], contre 77,7 % pour le hasard (p < 0,001).
    M1-texte donne 80,5 % (p = 0,115).
  - **Ne pas en conclure que M6 sélectionne mieux que M1-texte.** L'écart n'est pas testé, les IC
    se recouvrent, et le backtest est instable. Au bloc B, M1-texte faisait 85,3 % sur un pool de
    test qui ne diffère de celui-ci que de 16 projets (ceux sans date d'approbation connue). Le
    recalibrage n'y est pour rien : vérifié, même résultat avec la calibration du bloc B.
  - Avec un budget découpé par cohorte annuelle, quelques projets de plus ou de moins changent le
    portefeuille de ±5 points, soit l'ordre de grandeur de l'écart M6 − M1-texte.
  - En part du budget, M6 n'est pas significatif (85,3 %, p = 0,085).
- **Poids lisibles (régularisation forte : coefficients petits).** Associés à plus d'échecs :
  réinstallation de populations prévue (+0,04), nombre de cofinanceurs (+0,04), deuxième phase
  (+0,03), catégorie environnementale A (+0,03), capacité faible de l'agence (+0,03). Associés à
  moins d'échecs : catégorie B (−0,06). Ce sont des associations, pas des causes.
- **Limites.**
  - Exactitude des réponses mesurée seulement sur le pilote : 14/15 et 13/13 sur deux facteurs
    vérifiables ; first_operation exclu.
  - Pas de facteurs sur l'entraînement ≤ 2007 (quota Kaggle) : M5/M6 sont appris sur 853 projets.
  - Bloc D non évalué sur le coffre-fort, faute de textes.

# Bloc D : M5 et M6 — facteurs extraits par les agents (généré)

Choix de codage figés avant réglage : experiments/blocD/m6_choices.json.
Réglage : VC à 5 plis dans les 853 projets de validation (taux d'échec 22.7%). AUC-PR (VC) : M5 0.248 (C = 0.003), M6 0.327 (C = 0.003), texte seul 0.337.
**Test : 729 projets (taux d'échec 21.4%), calculé une seule fois.** Probabilités recalibrées sans le test.

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M0 (taux de base) | 0.214 | 0.500 | 0.1684 | 0.013 |
| M1-texte (Platt) | 0.297 [0.247 ; 0.372] | 0.604 [0.553 ; 0.653] | 0.1647 | 0.023 |
| M5 (facteurs seuls) | 0.241 [0.204 ; 0.293] | 0.566 [0.517 ; 0.614] | 0.1678 | 0.012 |
| M6 (texte + facteurs) | 0.300 [0.248 ; 0.369] | 0.601 [0.554 ; 0.649] | 0.1644 | 0.015 |

| Écart apparié | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| m6 - texte | 0.003 [-0.039 ; 0.043] | -0.002 [-0.041 ; 0.036] | -0.000 [-0.004 ; 0.003] |
| m5 - m0 | 0.027 [0.003 ; 0.070] | 0.066 [0.017 ; 0.114] | -0.001 [-0.001 ; -0.000] |

## Backtest (objectif figé Σ p × coût, budget 25%, ≤ 25 % par pays)

| Probabilités | projets | % réussis [IC 95 %] | p vs hasard | % budget → réussis [IC 95 %] | p vs hasard |
| --- | ---: | --- | ---: | --- | ---: |
| m6 | 180 | 86.1% [80.6% ; 91.1%] | 0.000 | 85.3% [77.5% ; 92.5%] | 0.085 |
| texte | 149 | 80.5% [73.8% ; 86.6%] | 0.115 | 81.2% [71.5% ; 89.9%] | 0.412 |
| hasard (1 000 tirages) | 224 | 77.7% | — | 80.3% | — |

## Poids (coefficients standardisés ; positif = plus d'échec)

- m5 : env_B -0.064, resettlement +0.041, follow_on_phase_missing -0.041, cofinanciers +0.040, env_category_missing +0.037, follow_on_phase +0.033, env_A +0.028, weak_capacity +0.028, political_risk_missing -0.027, implementing_agencies_missing -0.024, weak_capacity_missing -0.022, env_C +0.022, baseline_values +0.021, pilot_innovative +0.019, env_FI -0.017, components_missing -0.017, political_risk +0.016, pilot_innovative_missing -0.015, counterpart_funding_missing +0.013, reform_to_adopt +0.013, fragility_conflict +0.013, fragility_conflict_missing -0.012, cofinanciers_missing -0.010, overall_risk_rating_missing -0.009, decentralized_implementation -0.008, lessons_problems -0.008, fiduciary_risk_missing +0.007, implementing_agencies +0.006, reform_to_adopt_missing +0.006, components -0.004, economic_return_reported_missing +0.004, fiduciary_risk -0.003, decentralized_implementation_missing +0.003, overall_risk_rating -0.003, resettlement_missing +0.003, lessons_problems_missing +0.002, counterpart_funding +0.002, economic_return_reported +0.001, baseline_values_missing -0.001
- m6 : logit_text +0.159, env_B -0.070, env_category_missing +0.042, resettlement +0.039, follow_on_phase_missing -0.037, cofinanciers +0.033, follow_on_phase +0.032, env_A +0.030, political_risk_missing -0.024, implementing_agencies_missing -0.024, baseline_values +0.023, env_C +0.021, weak_capacity +0.021, pilot_innovative +0.020, components_missing -0.016, pilot_innovative_missing -0.016, weak_capacity_missing -0.015, env_FI -0.015, counterpart_funding_missing +0.015, reform_to_adopt +0.013, political_risk +0.013, fiduciary_risk_missing +0.010, overall_risk_rating -0.010, lessons_problems -0.009, fragility_conflict +0.009, resettlement_missing +0.008, decentralized_implementation -0.007, fragility_conflict_missing -0.007, fiduciary_risk -0.007, overall_risk_rating_missing -0.007, reform_to_adopt_missing +0.006, cofinanciers_missing -0.005, economic_return_reported_missing +0.005, implementing_agencies +0.005, lessons_problems_missing +0.004, components -0.004, baseline_values_missing -0.003, decentralized_implementation_missing +0.002, counterpart_funding +0.001, economic_return_reported -0.000
