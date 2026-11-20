# Choix de l'objectif de sélection (bloc A.2, généré automatiquement)

Probabilités : M1-texte entraîné sur ≤ 2007, recalibré sur la validation. Plafond 25 % du
budget par pays. Critère de choix fixé à l'avance : moyenne sur 3 budgets de la part du
budget allant à des projets satisfaisants, **sur la validation**.

## 1. Validation 2008–2011 (848 projets) — sert au choix

| Objectif | Budget | Projets financés | % du budget → projets satisfaisants | % de projets satisfaisants |
| --- | --- | ---: | ---: | ---: |
| p | 10% | 429 | 79.2 % | 77.4 % |
| p | 25% | 617 | 78.1 % | 76.5 % |
| p | 50% | 756 | 80.4 % | 77.1 % |
| p_cost | 10% | 85 | 97.5 % | 96.5 % |
| p_cost | 25% | 146 | 95.3 % | 92.5 % |
| p_cost | 50% | 283 | 92.6 % | 88.0 % |
| p_cost_min | 10% | 85 | 97.5 % | 96.5 % |
| p_cost_min | 25% | 146 | 95.3 % | 92.5 % |
| p_cost_min | 50% | 282 | 92.6 % | 87.9 % |
| hasard | 10% | 132 | 81.0 % | 77.1 % |
| hasard | 25% | 249 | 83.0 % | 77.8 % |
| hasard | 50% | 446 | 81.1 % | 77.9 % |

Score de choix : p = 79.2 %, p_cost = 95.1 %, p_cost_min = 95.1 %. **Objectif figé : `p_cost`** (p minimum = 0.779).

## 2. Test 2012–2017 (728 projets) — calculé une seule fois

| Budget | Objectif | % budget → satisfaisants [IC 95 %] | p vs hasard | % projets satisfaisants [IC 95 %] | p vs hasard | Projets |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 10% | **p_cost (figé)** | 83.6 % [73.0 % ; 92.8 %] | 0.215 | 80.5 % [72.4 % ; 88.5 %] | 0.127 | 87 |
| 10% | p (initial, transparence) | 77.6 % [72.2 % ; 82.9 %] | 0.644 | 75.8 % [70.9 % ; 80.4 %] | 0.581 | 327 |
| 10% | hasard (1 000 tirages) | 79.4 % [68.9 % ; 88.1 %] | — | 76.5 % [70.0 % ; 82.9 %] | — | — |
| 10% | taux de base de la cohorte | 82.4 % | — | 78.0 % | — | — |
| 25% | **p_cost (figé)** | 82.7 % [73.0 % ; 90.9 %] | 0.262 | 85.3 % [79.3 % ; 90.7 %] | 0.001 | 150 |
| 25% | p (initial, transparence) | 78.3 % [74.1 % ; 82.6 %] | 0.713 | 76.0 % [72.1 % ; 79.7 %] | 0.698 | 488 |
| 25% | hasard (1 000 tirages) | 80.3 % [72.7 % ; 87.3 %] | — | 77.2 % [72.6 % ; 82.1 %] | — | — |
| 25% | taux de base de la cohorte | 82.4 % | — | 78.0 % | — | — |
| 50% | **p_cost (figé)** | 85.7 % [79.6 % ; 91.1 %] | 0.050 | 83.9 % [79.1 % ; 88.3 %] | 0.001 | 273 |
| 50% | p (initial, transparence) | 81.4 % [77.2 % ; 85.0 %] | 0.623 | 77.5 % [74.3 % ; 80.8 %] | 0.548 | 619 |
| 50% | hasard (1 000 tirages) | 82.1 % [77.7 % ; 86.1 %] | — | 77.7 % [74.9 % ; 80.4 %] | — | — |
| 50% | taux de base de la cohorte | 82.4 % | — | 78.0 % | — | — |

## 3. Contrôle : d'où vient le gain ?

Même objectif figé (`p_cost`), mêmes tirages au hasard, sur le test ; seules
les probabilités changent. Avec M0 (probabilité constante), seul le montant des projets compte.
Montant moyen d'un projet du test : 143 M USD.

| Budget | Probabilités | % budget → satisfaisants [IC 95 %] | p vs hasard | % projets satisfaisants [IC 95 %] | p vs hasard | Projets | Montant moyen (M USD) |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 10% | M0 (constante) | 74.7 % [65.6 % ; 83.2 %] | 0.826 | 74.2 % [67.6 % ; 80.2 %] | 0.749 | 182 | 57 |
| 10% | M1 | 83.6 % [71.6 % ; 93.9 %] | 0.212 | 82.3 % [73.4 % ; 91.1 %] | 0.044 | 79 | 132 |
| 10% | M1-texte | 83.6 % [73.0 % ; 92.8 %] | 0.215 | 80.5 % [72.4 % ; 88.5 %] | 0.127 | 87 | 120 |
| 10% | hasard (1 000) | 79.4 % | — | 76.5 % | — | — | — |
| 25% | M0 (constante) | 82.4 % [74.8 % ; 89.1 %] | 0.288 | 76.4 % [71.2 % ; 81.6 %] | 0.641 | 250 | 104 |
| 25% | M1 | 84.8 % [76.5 % ; 92.3 %] | 0.116 | 81.8 % [75.8 % ; 87.9 %] | 0.035 | 165 | 158 |
| 25% | M1-texte | 82.7 % [73.0 % ; 90.9 %] | 0.262 | 85.3 % [79.3 % ; 90.7 %] | 0.001 | 150 | 174 |
| 25% | hasard (1 000) | 80.3 % | — | 77.2 % | — | — | — |
| 50% | M0 (constante) | 78.4 % [71.8 % ; 84.3 %] | 0.948 | 75.1 % [70.5 % ; 79.5 %] | 0.966 | 390 | 134 |
| 50% | M1 | 84.8 % [78.8 % ; 90.3 %] | 0.112 | 82.0 % [77.7 % ; 86.2 %] | 0.001 | 305 | 171 |
| 50% | M1-texte | 85.7 % [79.6 % ; 91.1 %] | 0.050 | 83.9 % [79.1 % ; 88.3 %] | 0.001 | 273 | 191 |
| 50% | hasard (1 000) | 82.1 % | — | 77.7 % | — | — | — |

**Lecture (écrite à la main après le calcul ; elle ne modifie rien de figé).**
- **La taille seule n'explique pas le gain.** Avec des probabilités constantes (M0), l'objectif
  Σ p × coût remplit le budget sans regarder la qualité : il ne bat jamais le hasard (p = 0,75, 0,64,
  0,97 sur la part de projets satisfaisants).
- **Le gain vient des probabilités du modèle, pas spécifiquement du texte.** M1 (variables
  structurées) obtient presque le même gain que M1-texte (82 % de projets satisfaisants à 25 % et
  50 % du budget, contre 85 % et 84 % pour M1-texte) ; leurs intervalles se recouvrent largement,
  donc on ne peut pas dire que le texte fait mieux ici.
- En part du budget, aucune variante n'est nettement au-dessus du hasard (meilleur cas :
  M1-texte à 50 %, p = 0,050).
