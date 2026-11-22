# M7 : lecture (écrit à la main)

**Question : les sorties intermédiaires du comité (verdicts du critique, réécritures, texte des
arguments Contre) apportent-elles une information au-delà du texte ? Réponse : pas de preuve.**

- Test (une seule fois, 2e usage des 300 projets) : M7 − texte = +0,027 [−0,059 ; 0,112] en AUC-PR
  et +0,032 [−0,044 ; 0,110] en AUC-ROC. Brier identique (+0,001). L'écart est positif, mais
  l'intervalle contient largement zéro.
- La validation annonçait l'inverse : en validation croisée, la meilleure configuration de M7 fait
  0,345, contre 0,366 pour le texte seul. Le signe positif au test n'est donc pas confirmé par la
  validation. Il faut le lire comme du bruit tant qu'il n'est pas reproduit (par exemple sur le
  coffre-fort, si le comité y passait).
- M7 est mal calibré (ECE 0,088 contre 0,030) : il n'a pas été recalibré, faute de données
  indépendantes (les 300 projets de validation servent déjà à l'apprentissage).
- Les coefficients ne sont pas interprétables comme un jugement cohérent. Un argument Pour
  « strong » sur la faisabilité va avec PLUS d'échec (+0,49), un argument Contre « strong » sur la
  faisabilité avec MOINS d'échec (−0,43). C'est le signe d'un ajustement à du bruit (300 projets,
  13 variables + TF-IDF).
- Le réglage a gardé le TF-IDF des arguments Contre (poids 1). Sans TF-IDF, le meilleur score en
  validation croisée est 0,314.

# Bloc C+ : M7 — sorties intermédiaires du comité (généré)

Variables : arguments jugés « strong » par critère et position, part « unsupported »,
part réécrite au raffinement, TF-IDF des arguments Contre ; plus le logit de M1-texte.
Réglage par validation croisée à 5 plis dans les 300 projets de validation :
C = 3.0, poids du TF-IDF = 1.0 ; AUC-PR (VC) 0.345 contre 0.366 pour le texte seul (même VC).

**Test : 300 projets (taux d'échec 21.3%), calculé une seule fois. C'est le 2e usage de ces 300 projets de test (après M3 / M4).**

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | ECE |
| --- | --- | --- | ---: | ---: |
| Texte (M1-texte, Platt) | 0.294 [0.214 ; 0.396] | 0.569 [0.487 ; 0.651] | 0.1674 | 0.030 |
| M7 (texte + sorties du comité) | 0.321 [0.236 ; 0.427] | 0.601 [0.518 ; 0.686] | 0.1680 | 0.088 |

| Écart apparié | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| M7 − texte | 0.027 [-0.059 ; 0.112] | 0.032 [-0.044 ; 0.110] | 0.001 [-0.011 ; 0.012] |

Coefficients (variables standardisées, positif = plus d'échec) : logit_text +0.687, strong_pro_relevance -0.127, strong_contra_relevance +0.350, strong_pro_economic_efficiency +0.111, strong_contra_economic_efficiency -0.111, strong_pro_feasibility +0.488, strong_contra_feasibility -0.426, strong_pro_risk +0.434, strong_contra_risk +0.050, strong_pro_impact_sustainability -0.172, strong_contra_impact_sustainability -0.014, share_unsupported +0.014, share_rewritten -0.069
