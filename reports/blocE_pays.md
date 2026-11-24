<!-- Fichier généré par `uv run python -m investagents.report.build` depuis src/investagents/report/blocE_template.md : ne pas modifier à la main. -->
# Bloc E : ce qui prédit vraiment l'échec, et la part du pays

*Protocole écrit et commité avant tout résultat : `reports/blocE_pays_protocole.md` (commit
`ef29775`). Tous les chiffres sont lus dans `experiments/blocE/results.json` par
`report/metrics.py`. Mesure unique, descriptive, sur les 755 projets du test 2012–2017 qui
ont un texte ; le coffre-fort n'est pas touché.*

## La conclusion en une phrase

**Le signal vient surtout du contenu du dossier : sans aucun nom de pays ni adjectif de nationalité, M1-texte garde
presque toute sa performance (AUC-PR 0,295 [0,243 ; 0,367] contre 0,297 [0,245 ; 0,370], soit 3 %
du gain au-dessus du hasard perdu) ; mais les noms ne sont pas le seul moyen de reconnaître un pays,
et d'autres indices (capitales, régions, institutions, monnaies) restent dans le texte.**

## Les chiffres

| Mesure | Valeur [IC 95 %] |
| --- | --- |
| M1-texte d'origine : AUC-PR | 0,297 [0,245 ; 0,370] |
| M1-texte d'origine : AUC-ROC | 0,598 [0,549 ; 0,648] |
| M1-texte sans noms de pays ni adjectifs : AUC-PR | 0,295 [0,243 ; 0,367] |
| M1-texte sans noms de pays ni adjectifs : AUC-ROC | 0,600 [0,550 ; 0,649] |
| Écart apparié (sans pays − d'origine) : AUC-PR | −0,003 [−0,011 ; 0,004] |
| Écart apparié (sans pays − d'origine) : AUC-ROC | +0,001 [−0,005 ; 0,007] |
| Taux de base (hasard pour l'AUC-PR) | 0,216 |
| Part du gain au-dessus du hasard perdue sans les noms | 3 % |
| Verdict selon la règle fixée à l'avance | surtout du contenu du dossier |
| Expressions supprimées par dossier (moyenne) ; part des caractères | 97 ; 0,52 % |
| Contrôle : le code d'origine redonne les prédictions d'origine (écart maximal) | 0 (identiques) |
| Mots « pays » / « adjectif de pays » parmi les 100 mots les plus lourds | 16 / 1 (17 %) |
| Mots de pays parmi les 20 qui pèsent le plus vers l'échec / vers la réussite | 4 / 6 |

*L'IC de M1-texte d'origine diffère d'un millième de celui du bloc B (graine du bootstrap fixée
dans ce protocole, 41 au lieu de 11) : même modèle, mêmes prédictions.*

## Les mots qui pèsent le plus dans le modèle d'origine

Poids de la régression logistique (variables TF-IDF) : positif = vers l'échec, négatif = vers la
réussite. « pays » ou « adjectif de pays » selon la liste figée avant le calcul.

**Vers l'échec**

| # | Mot | Poids | Pays ? |
| ---: | --- | ---: | --- |
| 1 | subloans | +0,91 | — |
| 2 | frameworks | +0,84 | — |
| 3 | zaire | +0,84 | pays |
| 4 | subloan | +0,82 | — |
| 5 | algeria | +0,76 | pays |
| 6 | kenya | +0,73 | pays |
| 7 | concession | +0,73 | — |
| 8 | tanzania | +0,73 | pays |
| 9 | africa | +0,72 | — |
| 10 | contained | +0,71 | — |
| 11 | losses | +0,71 | — |
| 12 | sofide | +0,71 | — |
| 13 | des | +0,70 | — |
| 14 | files | +0,68 | — |
| 15 | livestock | +0,67 | — |
| 16 | metering | +0,67 | — |
| 17 | nairobi | +0,66 | — |
| 18 | smallholder | +0,66 | — |
| 19 | documented | +0,64 | — |
| 20 | forestry | +0,64 | — |

**Vers la réussite**

| # | Mot | Poids | Pays ? |
| ---: | --- | ---: | --- |
| 1 | china | −1,07 | pays |
| 2 | tax | −0,87 | — |
| 3 | thousands | −0,85 | — |
| 4 | thailand | −0,81 | pays |
| 5 | rs | −0,80 | — |
| 6 | contracting | −0,78 | — |
| 7 | emergency | −0,78 | — |
| 8 | pages | −0,72 | — |
| 9 | korea | −0,71 | pays |
| 10 | municipalities | −0,71 | — |
| 11 | beneficiary | −0,69 | — |
| 12 | armenia | −0,68 | pays |
| 13 | traffic | −0,68 | — |
| 14 | commune | −0,65 | — |
| 15 | chile | −0,65 | pays |
| 16 | alleviation | −0,64 | — |
| 17 | vietnam | −0,63 | pays |
| 18 | pavement | −0,63 | — |
| 19 | notes | −0,62 | — |
| 20 | communes | −0,62 | — |

À lire avec prudence : « africa » n'est pas étiqueté parce que la règle fixée à l'avance exclut les
régions, et « nairobi » (une capitale), « sofide » (sigle, semble-t-il, d'une institution financière de l'ex-Zaïre)
ou « rs » (probablement l'abréviation de roupies) ne sont pas des noms de pays mais trahissent quand même le pays. Supprimer
les noms ne supprime donc pas toute l'information « pays ».

## Les facteurs extraits les plus liés à l'échec

Sur les 853 projets de validation du bloc D : taux d'échec quand le facteur est
présent, contre absent ; effet en points avec IC à 95 %. Descriptif, non causal, non ajusté.

| Facteur (présent) | Projets | Échec si présent | Échec si absent | Effet [IC 95 %] | Poids dans M5 |
| --- | ---: | ---: | ---: | --- | ---: |
| Deuxième phase / suite d'un projet | 508 | 25,0 % | 19,4 % | +5,6 pts [−0,1 ; 11,4] | +0,033 |
| Catégorie environnementale A | 96 | 27,1 % | 22,2 % | +4,9 pts [−4,4 ; 14,2] | +0,028 |
| Projet pilote / innovant | 751 | 23,3 % | 18,6 % | +4,7 pts [−3,1 ; 12,5] | +0,019 |
| Capacité faible de l'agence | 547 | 24,3 % | 19,9 % | +4,4 pts [−1,6 ; 10,1] | +0,028 |
| Réinstallation de populations | 371 | 25,1 % | 21,0 % | +4,1 pts [−1,6 ; 9,9] | +0,041 |
| Risque politique cité | 573 | 24,1 % | 20,0 % | +4,1 pts [−1,6 ; 10,0] | +0,016 |
| Valeurs de référence des indicateurs | 610 | 23,8 % | 20,2 % | +3,6 pts [−2,4 ; 9,3] | +0,021 |
| Nombre de cofinanceurs (au-dessus de la médiane) | 251 | 25,1 % | 21,8 % | +3,3 pts [−3,0 ; 9,9] | +0,040 |
| Fragilité / conflit | 345 | 24,3 % | 21,7 % | +2,7 pts [−3,3 ; 8,3] | +0,013 |
| Réforme encore à adopter | 586 | 23,5 % | 21,0 % | +2,6 pts [−3,3 ; 8,7] | +0,013 |
| Leçons : projet antérieur en difficulté | 544 | 22,8 % | 22,7 % | +0,1 pts [−5,7 ; 5,8] | −0,008 |
| Nombre de composantes (au-dessus de la médiane) | 361 | 22,7 % | 22,8 % | 0,0 pts [−5,8 ; 6,0] | −0,004 |
| Rentabilité économique chiffrée | 552 | 22,6 % | 22,9 % | −0,3 pts [−6,4 ; 5,8] | +0,001 |
| Risque fiduciaire élevé ou substantiel | 254 | 22,4 % | 22,9 % | −0,4 pts [−6,3 ; 5,9] | −0,003 |
| Exécution décentralisée | 676 | 22,6 % | 23,2 % | −0,5 pts [−7,6 ; 6,5] | −0,008 |
| Fonds de contrepartie requis | 441 | 22,2 % | 23,3 % | −1,1 pts [−6,6 ; 4,5] | +0,002 |
| Risque global élevé ou substantiel | 254 | 21,7 % | 23,2 % | −1,6 pts [−7,9 ; 5,0] | −0,003 |
| Première opération (exclu des modèles) | 166 | 21,1 % | 23,1 % | −2,1 pts [−8,6 ; 5,2] | exclu |
| Nombre d'agences d'exécution (au-dessus de la médiane) | 186 | 20,4 % | 23,4 % | −3,0 pts [−9,7 ; 4,0] | +0,006 |

Facteurs dont l'effet, seul, a un intervalle qui exclut zéro : 0 sur
19. Chacun pèse peu ; c'est leur combinaison (M5) qui porte un signal faible mais réel.

## Ce que ça change pour ce que tu peux dire au prof

- Tu peux dire : **« Le texte ne prédit pas l'échec seulement parce qu'il nomme le pays : retirer
  tous les noms de pays et adjectifs de nationalité ne change presque rien à la performance. »**
- Tu ne peux pas dire : « le pays ne joue aucun rôle ». Les noms de pays sont parmi les mots les
  plus lourds du modèle d'origine, et d'autres indices du pays (capitales, régions, institutions)
  restent dans le texte : ils peuvent prendre le relais quand on retire les noms.
- La formulation juste : **le signal ne dépend pas des noms de pays ; une part de l'information
  géographique peut passer par d'autres mots.**

## Limites

1. **Les noms ne sont pas le pays.** La suppression retire les noms et adjectifs de la liste, pas
   les capitales, les régions, les monnaies, les institutions ni les noms de personnes.
2. **Liste imparfaite.** Les expressions de 3 lettres ou moins sont ignorées (« Lao ») ; les
   régions (« Africa ») sont exclues par la règle ; certains adjectifs sont aussi des noms de
   langue (« French »), ce qui retire un peu plus que le pays.
3. **Le test 2012–2017 est relu une fois de plus** (descriptif, sans réglage) ; le coffre-fort
   n'est pas utilisé, faute de textes.
4. **Facteurs : effets bruts sur 853 projets**, non ajustés les uns sur les autres ; les
   intervalles sont larges.
5. **Une seule mesure, un seul modèle (TF-IDF + régression logistique).** Un autre modèle de texte
   pourrait dépendre davantage du pays.
