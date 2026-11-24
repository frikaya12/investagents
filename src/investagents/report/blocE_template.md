# Bloc E : ce qui prédit vraiment l'échec, et la part du pays

*Protocole écrit et commité avant tout résultat : `reports/blocE_pays_protocole.md` (commit
`ef29775`). Tous les chiffres sont lus dans `experiments/blocE/results.json` par
`report/metrics.py`. Mesure unique, descriptive, sur les {{e_n_test}} projets du test 2012–2017 qui
ont un texte ; le coffre-fort n'est pas touché.*

## La conclusion en une phrase

**Le signal vient {{e_verdict}} : sans aucun nom de pays ni adjectif de nationalité, M1-texte garde
presque toute sa performance (AUC-PR {{e_nc_aucpr}} contre {{e_orig_aucpr}}, soit {{e_share_lost}}
du gain au-dessus du hasard perdu) ; mais les noms ne sont pas le seul moyen de reconnaître un pays,
et d'autres indices (capitales, régions, institutions, monnaies) restent dans le texte.**

## Les chiffres

| Mesure | Valeur [IC 95 %] |
| --- | --- |
| M1-texte d'origine : AUC-PR | {{e_orig_aucpr}} |
| M1-texte d'origine : AUC-ROC | {{e_orig_aucroc}} |
| M1-texte sans noms de pays ni adjectifs : AUC-PR | {{e_nc_aucpr}} |
| M1-texte sans noms de pays ni adjectifs : AUC-ROC | {{e_nc_aucroc}} |
| Écart apparié (sans pays − d'origine) : AUC-PR | {{e_diff_aucpr}} |
| Écart apparié (sans pays − d'origine) : AUC-ROC | {{e_diff_aucroc}} |
| Taux de base (hasard pour l'AUC-PR) | {{e_base}} |
| Part du gain au-dessus du hasard perdue sans les noms | {{e_share_lost}} |
| Verdict selon la règle fixée à l'avance | {{e_verdict}} |
| Expressions supprimées par dossier (moyenne) ; part des caractères | {{e_expr_per_doc}} ; {{e_chars_removed}} |
| Contrôle : le code d'origine redonne les prédictions d'origine (écart maximal) | {{e_repro}} |
| Mots « pays » / « adjectif de pays » parmi les 100 mots les plus lourds | {{e_top100_country}} / {{e_top100_adjective}} ({{e_top100_share}}) |
| Mots de pays parmi les 20 qui pèsent le plus vers l'échec / vers la réussite | {{e_top20_failure_country_or_adj}} / {{e_top20_success_country_or_adj}} |

*L'IC de M1-texte d'origine diffère d'un millième de celui du bloc B (graine du bootstrap fixée
dans ce protocole, 41 au lieu de 11) : même modèle, mêmes prédictions.*

## Les mots qui pèsent le plus dans le modèle d'origine

Poids de la régression logistique (variables TF-IDF) : positif = vers l'échec, négatif = vers la
réussite. « pays » ou « adjectif de pays » selon la liste figée avant le calcul.

**Vers l'échec**

@@table:e_words_fail@@

**Vers la réussite**

@@table:e_words_success@@

À lire avec prudence : « africa » n'est pas étiqueté parce que la règle fixée à l'avance exclut les
régions, et « nairobi » (une capitale), « sofide » (sigle, semble-t-il, d'une institution financière de l'ex-Zaïre)
ou « rs » (probablement l'abréviation de roupies) ne sont pas des noms de pays mais trahissent quand même le pays. Supprimer
les noms ne supprime donc pas toute l'information « pays ».

## Les facteurs extraits les plus liés à l'échec

Sur les {{e_n_valid_factors}} projets de validation du bloc D : taux d'échec quand le facteur est
présent, contre absent ; effet en points avec IC à 95 %. Descriptif, non causal, non ajusté.

@@table:e_factors@@

Facteurs dont l'effet, seul, a un intervalle qui exclut zéro : {{e_n_factors_sig}} sur
{{e_n_factors}}. Chacun pèse peu ; c'est leur combinaison (M5) qui porte un signal faible mais réel.

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
