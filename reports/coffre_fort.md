# Coffre-fort : lecture (écrit à la main)

**Chiffre principal (tout le coffre, 700 projets) : M1 ne prédit plus rien.** AUC-PR 0,172
[0,144 ; 0,211] pour un taux de base de 0,176 ; AUC-ROC 0,509 [0,457 ; 0,562] ; M1 − M0 = −0,004
[−0,019 ; 0,021]. C'est vrai pour les projets approuvés ≤ 2017 (0,504) comme ≥ 2018 (0,515). Le
signal faible de M1 au test 2012–2017 (AUC-ROC 0,567) ne se retrouve pas sur des données neuves.
Raisons possibles : la dérive (taux d'échec 17,6 % ici contre 28 % à l'entraînement ; M0 lui-même
est mal calibré, ECE 0,108), des projets plus récents que l'entraînement (≤ 2007), et un test
2012–2017 consulté plusieurs fois en phase 2, donc optimiste.

**M1-texte : descriptif seulement (amendement).** Sur les 78 projets dont le texte avait pu être
téléchargé (12 échecs seulement), M1-texte fait mieux que M1 sur les mêmes projets : AUC-PR 0,259
contre 0,137, écart +0,122 [0,002 ; 0,298], et Brier meilleur. Mais l'AUC-ROC de M1-texte est
0,543 [0,376 ; 0,724], compatible avec le hasard. Avec 12 échecs et un sous-échantillon non
aléatoire (72 des 78 projets approuvés ≤ 2017), ce chiffre ne permet pas de conclure que le texte
garde son signal sur le coffre.

**Conséquence pour le projet :** les chiffres du test 2012–2017 surestiment ce que les modèles sans
agents font sur des données vraiment neuves. Le rapport final le dit.

# Coffre-fort : évaluation finale unique, modèles sans agents (généré)

Protocole figé avant lecture : experiments/vault/protocol.json. Modèles entraînés sur
≤ 2007, recalibrés (Platt) sur 2008–2011, appliqués une seule fois aux projets évalués
par l'IEG en exercice ≥ 2025 (700 projets, dont 78
avec un dossier ex ante exploitable). Classe positive = échec.

**Amendement (experiments/vault/protocol_amendment.json, commité avant lecture).** Le
serveur de documents a refusé (403) presque tous les téléchargements : M0 et M1 portent
sur tout le coffre (chiffres principaux). M1-texte ne porte que sur les projets dont le
dossier était déjà téléchargé : **résultat descriptif**, sous-échantillon non aléatoire
(ordre de téléchargement, refus 403) et petit ; M1 est donné sur le même
sous-échantillon.

## tout le coffre-fort (700 projets, taux d'échec 17.6%)

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M0 | 0.176 (= taux d'échec) | 0.500 | 0.1565 | 0.108 |
| M1 | 0.172 [0.144 ; 0.211] | 0.509 [0.457 ; 0.562] | 0.1545 | 0.073 |

M1 − M0 : AUC-PR -0.004 [-0.019 ; 0.021], Brier -0.002 [-0.007 ; 0.003].

Projets avec texte téléchargé (78, taux d'échec 15.4%) — descriptif :

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M1 | 0.137 [0.078 ; 0.258] | 0.411 [0.235 ; 0.592] | 0.1476 | 0.086 |
| M1-texte | 0.259 [0.105 ; 0.502] | 0.543 [0.376 ; 0.724] | 0.1360 | 0.055 |

M1-texte − M1 : AUC-PR 0.122 [0.002 ; 0.298], AUC-ROC 0.132 [-0.020 ; 0.281], Brier -0.012 [-0.023 ; -0.002].

## approbation ≤ 2017 (286 projets, taux d'échec 14.0%)

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M0 | 0.140 (= taux d'échec) | 0.500 | 0.1410 | 0.144 |
| M1 | 0.140 [0.101 ; 0.202] | 0.504 [0.411 ; 0.599] | 0.1371 | 0.113 |

M1 − M0 : AUC-PR 0.000 [-0.021 ; 0.047], Brier -0.004 [-0.011 ; 0.004].

Projets avec texte téléchargé (72, taux d'échec 15.3%) — descriptif :

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M1 | 0.140 [0.073 ; 0.261] | 0.428 [0.250 ; 0.611] | 0.1464 | 0.092 |
| M1-texte | 0.270 [0.092 ; 0.508] | 0.559 [0.375 ; 0.736] | 0.1350 | 0.060 |

M1-texte − M1 : AUC-PR 0.129 [-0.006 ; 0.321], AUC-ROC 0.130 [-0.041 ; 0.290], Brier -0.011 [-0.023 ; 0.000].

## approbation ≥ 2018 (414 projets, taux d'échec 20.0%)

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M0 | 0.200 (= taux d'échec) | 0.500 | 0.1672 | 0.083 |
| M1 | 0.201 [0.161 ; 0.257] | 0.515 [0.445 ; 0.583] | 0.1666 | 0.065 |

M1 − M0 : AUC-PR 0.000 [-0.022 ; 0.038], Brier -0.001 [-0.007 ; 0.006].

Projets avec texte téléchargé : 6 — trop peu de projets ou d'échecs pour des métriques.

