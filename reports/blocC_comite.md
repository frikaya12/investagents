# Bloc C : comité complet sur Kaggle — fonctionnement, citations, dispersion (généré)

Qwen2.5 7B (Ollama), 2 GPU T4 Kaggle, débat Pour / Contre → critique → raffinement
(1 tour) → Juge à grille ancrée. 300 projets de validation + 300 du test, ordre alterné.
**M3 / M4 non calculés ; issue du test non regardée** (en attente de validation).

## 1. Fonctionnement

- Projets notés : 600 / 600 (validation 300, test 300) ; projets avec au moins une erreur consignée : 0.
- Temps par projet (un GPU) : médiane 89 s, 90e centile 97 s, max 161 s ; 2 GPU en parallèle ; durée totale 7.5 h (quota gratuit Kaggle). Coût : 0 €.
- Jetons : sortie médiane 2228 par projet ; plus long prompt 6253 jetons (fenêtre 12 288 : aucune troncature).

## 2. Citations

- Numéros de phrases existants (contrôle sur Kaggle) : 99.8% des arguments en moyenne.
- Vérification mot pour mot dans le texte complet masqué (refaite en local) : 5986 / 6000 = 99.8%.

## 3. Dispersion des notes (0–10), toutes périodes

| Critère | moyenne | écart-type | min | max | part ≥ 9 | part = 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relevance | 7.79 | 0.65 | 5 | 8 | 0.0% | 0.0% |
| economic_efficiency | 6.51 | 1.26 | 2 | 8 | 0.0% | 0.0% |
| feasibility | 6.46 | 1.21 | 4 | 8 | 0.0% | 0.0% |
| risk | 7.54 | 0.88 | 5 | 9 | 0.5% | 0.0% |
| impact_sustainability | 6.52 | 1.21 | 5 | 8 | 0.0% | 0.0% |
| mean_score | 6.96 | 0.64 | 5 | 8 | 0.0% | 0.0% |

Corrélation moyenne entre critères : 0.21 (min -0.00, max 0.39). Notes distinctes utilisées : [2, 3, 4, 5, 6, 7, 8, 9].

## 4. Critique et raffinement

- Verdicts du critique : strong 3029, weak 2926, unsupported 40 (sur 5995).
- Part des arguments réécrits au raffinement : 41%.

## 5. Test de reconnaissance (50 projets, réponses comparées en local)

- Se dit « reconnu » : 0%.
- Pays correct : 86% (le pays est souvent écrit dans le texte : ce n'est pas une preuve de mémorisation).
- Année d'approbation à ±1 an : 36%.
- Nom du projet : similarité médiane 0.14, ≥ 0,8 pour 1 projet(s).
- Note IEG proposée (≠ « unknown ») : 0 / 50.

## 6. Lien notes / échec sur la VALIDATION seulement (300 projets, taux d'échec 22.7%) — diagnostic

Une note élevée doit aller avec MOINS d'échecs : on mesure l'AUC du score « 10 − note »
pour prédire l'échec (0,5 = hasard).

| Score | AUC-ROC | AUC-PR (taux de base 0.227) |
| --- | ---: | ---: |
| relevance | 0.495 | 0.226 |
| economic_efficiency | 0.496 | 0.226 |
| feasibility | 0.534 | 0.247 |
| risk | 0.517 | 0.230 |
| impact_sustainability | 0.417 | 0.204 |
| mean_score | 0.482 | 0.223 |

Pas d'intervalle de confiance ici : diagnostic sur 300 projets. Les chiffres qui
comptent (M3, M4 avec IC) seront réglés sur la validation, puis test une seule fois.
