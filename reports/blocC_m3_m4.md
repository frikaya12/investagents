# Bloc C : lecture (écrit à la main)

**Réponse à la question de recherche, sur cet échantillon : non.** Les 5 notes du comité
(Qwen2.5 7B, débat → critique → raffinement → Juge à grille ancrée) n'apportent pas d'information
utile pour prédire l'échec des projets, ni seules ni en plus du texte.

- M3 (5 notes) : AUC-PR 0,200 au test pour un taux de base de 0,213 ; M3 − M0 = −0,013
  [−0,037 ; 0,046]. Niveau du hasard.
- M4 (texte + notes) fait moins bien que le texte seul : −0,031 [−0,088 ; 0,029] en AUC-PR (non
  significatif) et un Brier plus mauvais de 0,007 [0,000 ; 0,015]. Les notes ajoutent du bruit.
- Sur la validation, la validation croisée laissait espérer un petit gain (M4 0,398 contre 0,374
  pour le texte seul). Il ne se retrouve pas au test : c'était du sur-apprentissage sur
  300 projets.
- Poids de M3 : le seul poids notable est celui de impact_sustainability, qui est positif (une
  meilleure note va avec PLUS d'échecs). Ce sens contre-intuitif était déjà visible sur la
  validation (AUC 0,417) et ne se transfère pas.
- Raisons probables :
  - les notes sont très resserrées (écart-type de 0,65 à 1,26 ; la note médiane de relevance est
    presque toujours 8) ;
  - un 7B lit des dossiers d'évaluation ex ante, rédigés pour être approuvés : ils exposent peu
    les faiblesses qui feront échouer le projet ;
  - 300 projets d'apprentissage seulement.

**Limites :**
1. Le masquage retire le nom complet du projet et ses identifiants, mais laisse les sigles et
   certaines variantes du nom (par exemple « FMR 2 », « MPHEQIP », « Lebanon Fiscal Management
   Reform Project (LFMR) »). Le test de reconnaissance ne montre pas de mémorisation : 0/50
   « reconnu », 0/50 note IEG proposée, et des notes non prédictives sur la validation (une issue
   mémorisée les rendrait prédictives). Décision d'Aya : pas de nouveau passage du comité.
2. Un seul modèle (7B quantifié), un seul passage, 1 tour de raffinement. Les ablations (sans
   critique, 2 tours) et la stabilité sur 3 passages n'ont pas été faites faute de quota.
3. M4 est un empilement logistique, pas le gradient boosting du cahier des charges : un gradient
   boosting sur 300 projets sur-apprendrait.
4. Les IC du test sont larges (n = 300, 64 échecs) : un petit apport (moins de 0,05 en AUC-PR)
   ne peut pas être exclu.

# Bloc C : M3 et M4 — apport des notes du comité (généré)

Réglage : validation croisée à 5 plis dans les 300 projets de validation
(taux d'échec 22.7%). Test : 300 projets, taux
d'échec 21.3%, calculé une seule fois. Classe positive = échec.

| Modèle | C | AUC-PR validation (VC) |
| --- | ---: | ---: |
| m3 | 3.0 | 0.277 |
| m4 | 1.0 | 0.398 |

## Test (une seule fois)

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| M0 (taux de base) | 0.213 (= taux d'échec) | 0.500 | 0.1680 | 0.013 |
| Texte (M1-texte figé, Platt) | 0.294 [0.214 ; 0.396] | 0.569 [0.487 ; 0.651] | 0.1674 | 0.030 |
| M3 (5 notes) | 0.200 [0.152 ; 0.272] | 0.472 [0.397 ; 0.549] | 0.1751 | 0.059 |
| M4 (texte + 5 notes) | 0.263 [0.193 ; 0.361] | 0.544 [0.460 ; 0.625] | 0.1747 | 0.076 |

| Écart apparié (même tirages) | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| m4 - texte | -0.031 [-0.088 ; 0.029] | -0.025 [-0.074 ; 0.023] | 0.007 [0.000 ; 0.015] |
| m3 - m0 | -0.013 [-0.037 ; 0.046] | -0.028 [-0.103 ; 0.049] | 0.007 [0.001 ; 0.014] |
| m3 - texte | -0.094 [-0.185 ; -0.009] | -0.097 [-0.206 ; 0.012] | 0.008 [-0.003 ; 0.018] |

## Poids appris (coefficients sur variables standardisées ; positif = plus d'échec)

- m3 : relevance -0.025, economic_efficiency -0.109, feasibility -0.216, risk +0.007, impact_sustainability +0.411, intercept -1.264
- m4 : logit_texte +0.643, relevance -0.053, economic_efficiency -0.157, feasibility -0.211, risk -0.001, impact_sustainability +0.468, intercept -1.369
