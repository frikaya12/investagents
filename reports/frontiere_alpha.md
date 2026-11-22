# Frontière α : lecture (écrit à la main)

**Question : combien de pertinence et d'impact les agents ajoutent-ils au portefeuille, et pour
quel coût en taux de réussite ?** Univers : les ~300 projets notés par le comité par période,
budget 25 %, poids égaux, α fixés à l'avance.

- **Ce que les agents ajoutent (selon leurs propres notes).** De α = 1 (ML seul) à α = 0 (comité
  seul), la note moyenne du portefeuille monte de +1,15 point d'impact sur la validation et de
  +0,75 sur le test. La pertinence ne monte que de +0,20 et +0,14 : elle est presque constante
  (écart-type 0,65, la note 8 domine). L'efficience et la faisabilité montent aussi (+0,6 à +0,9).
  **Attention :** ce gain est mesuré avec les notes qui servent à choisir. Il est donc vrai par
  construction. Il dit que le portefeuille correspond mieux au jugement du comité, pas que les
  projets auront plus d'impact. Aucune donnée externe ne valide ces notes (elles ne prédisent pas
  l'issue IEG).
- **Coût en réussite.**
  - Validation : la part de projets réussis passe de 91,5 % (α = 1) à 78,6 % (α = 0), soit −13
    points. Le comité seul revient au niveau du hasard (77,0 %). La réussite attendue (moyenne des
    probabilités du ML) baisse de 10 points.
  - Mais la validation favorise α = 1 : M1-texte a été choisi, et ses probabilités recalibrées,
    sur ces mêmes années.
  - Test (une seule fois, descriptif) : les écarts sont faibles et dans le bruit. 79,8 % (α = 1),
    82,0 % (0,75), 83,5 % (0,5), 79,3 % (0,25), 78,6 % (0), hasard 76,7 %. Les IC se recouvrent
    tous (largeur environ ±8 points, ~85 projets par portefeuille). La réussite attendue baisse
    de 6 points de α = 1 à α = 0.
- **Lecture honnête.** Sur ces échantillons, un α intermédiaire (0,5 à 0,75) donne un portefeuille
  mieux noté par le comité, sans perte de réussite mesurable au test. Sur la validation, la perte
  est nette (−6 à −9 points à α = 0,75 et 0,5). Le test ne permet pas de distinguer les α ; aucun
  α n'est recommandé à partir du test.
- Sur ce sous-échantillon de 295 projets, α = 1 au test (79,8 %) fait moins bien que le
  portefeuille figé sur les 728 projets du test (85,3 %, blocB_resultats.md). Le pool est plus
  petit (~50 projets par cohorte au lieu de ~120) : le sac à dos a moins de choix.

# Frontière α : ML + agents dans la décision multicritère (généré)

score = α · p_succès (M1-texte calibré) + (1 − α) · moyenne des 5 notes du comité / 10
(poids égaux). Sac à dos `p_cost` par cohorte annuelle, budget 25%,
au plus 25% du budget par pays.
α fixés à l'avance, aucun n'est choisi.
Notes moyennes du portefeuille sur 10 ; « hasard » = moyenne de 1 000 portefeuilles.

### Validation 2008–2011 (297 projets, taux de réussite 77.8%)

| α | projets | réussis [IC 95 %] | budget → réussis | réussite attendue | pert. | effic. | faisab. | risque | impact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 71 | 91.5% [84.5% ; 97.2%] | 94.9% | 87.0% | 7.77 | 6.51 | 6.34 | 7.62 | 6.35 |
| 0.75 | 74 | 85.1% [77.0% ; 93.2%] | 93.4% | 86.6% | 7.82 | 6.77 | 6.59 | 7.69 | 6.74 |
| 0.50 | 82 | 82.9% [74.4% ; 90.2%] | 88.6% | 84.3% | 7.93 | 7.16 | 6.99 | 7.84 | 7.09 |
| 0.25 | 82 | 78.0% [68.3% ; 86.6%] | 85.8% | 81.0% | 7.98 | 7.44 | 7.15 | 7.91 | 7.54 |
| 0.00 | 98 | 78.6% [70.4% ; 86.7%] | 84.3% | 77.0% | 7.98 | 7.44 | 7.21 | 7.93 | 7.50 |
| hasard | 116 | 77.0% | 83.2% | 76.4% | 7.81 | 6.54 | 6.41 | 7.64 | 6.49 |

Écart-type dans l'univers : p_succès 0.094, score du comité 0.061.
De α = 1 à α = 0 : pertinence +0.20, impact +1.15 (points sur 10) ; part de projets réussis -13.0 points ; réussite attendue -10.0 points.

### Test 2012–2017 (une seule fois, descriptif) (295 projets, taux de réussite 78.6%)

| α | projets | réussis [IC 95 %] | budget → réussis | réussite attendue | pert. | effic. | faisab. | risque | impact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 84 | 79.8% [71.4% ; 88.1%] | 80.7% | 87.3% | 7.83 | 6.62 | 6.71 | 7.55 | 6.65 |
| 0.75 | 89 | 82.0% [74.2% ; 89.9%] | 81.9% | 86.5% | 7.92 | 6.94 | 6.99 | 7.67 | 6.85 |
| 0.50 | 85 | 83.5% [75.3% ; 91.8%] | 82.9% | 85.6% | 7.93 | 7.05 | 7.12 | 7.80 | 7.15 |
| 0.25 | 92 | 79.3% [70.7% ; 87.0%] | 81.9% | 82.5% | 7.97 | 7.21 | 7.36 | 7.83 | 7.34 |
| 0.00 | 98 | 78.6% [70.4% ; 85.7%] | 78.8% | 81.2% | 7.97 | 7.19 | 7.45 | 7.82 | 7.41 |
| hasard | 125 | 76.7% | 81.9% | 79.2% | 7.75 | 6.42 | 6.51 | 7.43 | 6.49 |

Écart-type dans l'univers : p_succès 0.081, score du comité 0.068.
De α = 1 à α = 0 : pertinence +0.14, impact +0.75 (points sur 10) ; part de projets réussis -1.2 points ; réussite attendue -6.1 points.

