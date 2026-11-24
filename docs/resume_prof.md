<!-- Fichier généré par `uv run python -m investagents.report.build` depuis src/investagents/report/resume_template.md : ne pas modifier à la main. -->
# InvestAgents : résumé de présentation

**Le projet en bref.** Données : notes de résultat IEG (Banque mondiale) et dossiers d'évaluation
écrits avant l'approbation, 8 260 projets. Anti-fuite : documents datés avant l'approbation,
variables post-projet interdites, noms masqués. Découpage temporel : entraînement ≤ 2007, validation
2008–2011, test 2012–2017, coffre-fort (évalués en 2025 ou après) ouvert une seule fois. Modèles : M0
(taux de base), M1 (gradient boosting), M1-texte (TF-IDF) ; M3, M4, M7 sur les sorties du comité
d'agents (Qwen2.5 7B via Ollama : Pour / Contre, critique, Juge sur 5 critères) ; M5, M6 sur 19
facteurs de risque extraits avec citations (M2 non fait). Sélection : sac à dos CP-SAT sous budget et
frontière α entre probabilité de réussite et notes du comité.

| Étape | Métrique (IC 95 %) | Verdict | En clair |
| --- | --- | --- | --- |
| Données | 8 260 projets exploitables (seuil 300) | Bon | Assez de vrais projets pour apprendre et tester. |
| Anti-fuite | 0 / 50 projet reconnu par le LLM ; colonnes post-projet bloquées par un test | Bon | Les agents ne trichent pas avec leur mémoire. |
| M0 · taux de base | AUC-PR 0,217 (test) | Référence | Le score du hasard, le plancher à battre. |
| M1 · gradient boosting | AUC-PR 0,247 [0,216 ; 0,286] ; AUC-ROC 0,567 [0,529 ; 0,603] | Faible | Les variables chiffrées battent à peine le hasard ; seule la région compte. |
| M1-texte · TF-IDF | AUC-PR 0,297 [0,245 ; 0,369] contre 0,216 ; écart avec M1 +0,039 [0,003 ; 0,082] | Faible mais réel | Le meilleur modèle : le texte du dossier prédit un peu l'échec. |
| Bloc E · part du pays | M1-texte sans noms de pays ni adjectifs : AUC-PR 0,295 [0,243 ; 0,367] contre 0,297 [0,245 ; 0,370] ; écart −0,003 [−0,011 ; 0,004] | Contenu du dossier | Retirer les noms de pays ne change presque rien ; mais d'autres mots (capitales, régions) trahissent encore le pays. |
| Coffre-fort · M1 | AUC-PR 0,172 [0,144 ; 0,211] contre 0,176 ; AUC-ROC 0,509 [0,457 ; 0,562] | Négatif | Sur des projets neufs, M1 ne prédit plus rien : niveau du hasard. |
| Coffre-fort · M1-texte | écart avec M1 +0,122 [0,002 ; 0,298] (78 projets) | Non concluant | Trop peu de textes téléchargés (refus du serveur) pour conclure. |
| Comité · fiabilité | 600 / 600 projets, 0 erreur ; citations exactes 99,8 % | Bon | Le comité tourne à l'échelle et cite vraiment le dossier. |
| M3 · notes du Juge | AUC-PR 0,200 [0,152 ; 0,272] contre 0,213 | Négatif | Les notes des juges ne prédisent pas l'échec. |
| M4 · texte + notes | écart avec le texte −0,031 [−0,088 ; 0,029] | Négatif | Ajouter les notes au texte n'apporte rien. |
| M7 · texte + débat | écart avec le texte +0,027 [−0,059 ; 0,112] | Non démontré | Zéro est dans l'intervalle ; la validation annonçait l'inverse. |
| Variables d'entrée | secteur manquant pour 78 % des projets d'avant 1996 ; pas de rupture de nomenclature en 2014 (plus grand changement de part : 2,2 points) ; catégorie environnementale et secteurs de l'API non utilisés | Limite | Le secteur pèse zéro dans M1 ; des variables utiles restent à tester. |
| Comité · qualité du débat | Pour et Contre citent les mêmes phrases dans 5 % des paires (158 sur 3 000) | Limite | Une citation exacte ne garantit pas un argument pertinent. |
| Extraction · fiabilité | 1 582 dossiers, 0 erreur ; citations valides 97,6 % | Bon | L'extracteur lit les dossiers à l'échelle avec des citations vérifiées. |
| M5 · 19 facteurs | écart avec M0 +0,027 [0,003 ; 0,070] | Faible mais réel | Les facteurs extraits portent un vrai signal, petit. |
| M6 · texte + facteurs | écart avec le texte +0,003 [−0,039 ; 0,043] | Négatif | Ce signal est déjà dans le texte complet. |
| Sélection CP-SAT | 85,3 % [79,3 ; 90,7] de projets réussis (25 % du budget) contre 77,2 % au hasard (p ≤ 0,001) | Bon | Le portefeuille choisi réussit mieux que le hasard. |
| Contrôle M0 | 76,4 % (p = 0,641) | Contrôle | Choisir sur la seule taille ne bat pas le hasard : le gain vient du ML. |
| Fragilité du backtest | même modèle : 85,3 % (bloc B) contre 80,5 % (bloc D), 16 projets d'écart | Fragile | ± 5 points selon le pool : un écart de cette taille entre modèles ne veut rien dire. |
| Frontière α | test : 79,8 % (α = 1), 83,5 % (α = 0,5), 78,6 % (α = 0) ; impact 6,65 → 7,41 ; validation −13,0 points | Arbitrage | Plus de poids au comité : portefeuille mieux noté, mais coût en réussite en validation. |

**AUC-ROC au test, IC à 95 %** (0,5 = hasard, quel que soit le taux d'échec). Chaque modèle est
mesuré sur son propre échantillon de test : on compare d'abord à 0,5, puis au texte seul sur le
même échantillon.

| Modèle | Échantillon de test | AUC-ROC [IC 95 %] | IC entièrement au-dessus de 0,5 ? |
| --- | --- | --- | --- |
| M0 · taux de base | test complet (n = 1 294) | 0,500 [0,500 ; 0,500] | — (hasard par construction) |
| M1 · gradient boosting | test complet (n = 1 294) | 0,567 [0,529 ; 0,603] | oui |
| M1 · gradient boosting | projets avec texte (n = 755) | 0,573 [0,523 ; 0,621] | oui |
| M1-texte · TF-IDF | projets avec texte (n = 755) | 0,598 [0,546 ; 0,647] | oui |
| Texte seul (référence du comité) | test du comité (n = 300) | 0,569 [0,487 ; 0,651] | non |
| M3 · 5 notes du Juge | test du comité (n = 300) | 0,472 [0,397 ; 0,549] | non |
| M4 · texte + notes | test du comité (n = 300) | 0,544 [0,460 ; 0,625] | non |
| M7 · texte + débat | test du comité (n = 300) | 0,601 [0,518 ; 0,686] | oui |
| Texte seul (référence du bloc D) | test du bloc D (n = 729) | 0,604 [0,553 ; 0,653] | oui |
| M5 · 19 facteurs | test du bloc D (n = 729) | 0,566 [0,517 ; 0,614] | oui |
| M6 · texte + facteurs | test du bloc D (n = 729) | 0,601 [0,554 ; 0,649] | oui |
| M1 · coffre-fort | projets évalués en 2025+ (n = 700) | 0,509 [0,457 ; 0,562] | non |

*Lecture : les AUC-ROC restent modestes (au mieux 0,60). M3 est sous 0,5 et son IC
contient 0,5 : les notes du Juge ne classent pas mieux que le hasard. Le texte seul du comité a un
IC qui touche 0,5, faute de projets (300). Sur le coffre-fort, M1 tombe à 0,5. Ce tableau ne dit
pas si un modèle bat un autre : pour cela, voir les écarts appariés.*

**Conclusion.** Le ML prédit faiblement ; les agents justifient chaque décision avec des sources
vérifiées et extraient des facteurs de risque lisibles et prédictifs, mais n'ajoutent rien au texte
complet.
