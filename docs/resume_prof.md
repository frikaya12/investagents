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
| Coffre-fort · M1 | AUC-PR 0,172 [0,144 ; 0,211] contre 0,176 ; AUC-ROC 0,509 [0,457 ; 0,562] | Négatif | Sur des projets neufs, M1 ne prédit plus rien : niveau du hasard. |
| Coffre-fort · M1-texte | écart avec M1 +0,122 [0,002 ; 0,298] (78 projets) | Non concluant | Trop peu de textes téléchargés (refus du serveur) pour conclure. |
| Comité · fiabilité | 600 / 600 projets, 0 erreur ; citations exactes 99,8 % | Bon | Le comité tourne à l'échelle et cite vraiment le dossier. |
| M3 · notes du Juge | AUC-PR 0,200 [0,152 ; 0,272] contre 0,213 | Négatif | Les notes des juges ne prédisent pas l'échec. |
| M4 · texte + notes | écart avec le texte −0,031 [−0,088 ; 0,029] | Négatif | Ajouter les notes au texte n'apporte rien. |
| M7 · texte + débat | écart avec le texte +0,027 [−0,059 ; 0,112] | Non démontré | Zéro est dans l'intervalle ; la validation annonçait l'inverse. |
| Comité · qualité du débat | Pour et Contre citent les mêmes phrases dans 5 % des paires (158 sur 3 000) | Limite | Une citation exacte ne garantit pas un argument pertinent. |
| Extraction · fiabilité | 1 582 dossiers, 0 erreur ; citations valides 97,6 % | Bon | L'extracteur lit les dossiers à l'échelle avec des citations vérifiées. |
| M5 · 19 facteurs | écart avec M0 +0,027 [0,003 ; 0,070] | Faible mais réel | Les facteurs extraits portent un vrai signal, petit. |
| M6 · texte + facteurs | écart avec le texte +0,003 [−0,039 ; 0,043] | Négatif | Ce signal est déjà dans le texte complet. |
| Sélection CP-SAT | 85,3 % [79,3 ; 90,7] de projets réussis (25 % du budget) contre 77,2 % au hasard (p ≤ 0,001) | Bon | Le portefeuille choisi réussit mieux que le hasard. |
| Contrôle M0 | 76,4 % (p = 0,641) | Contrôle | Choisir sur la seule taille ne bat pas le hasard : le gain vient du ML. |
| Fragilité du backtest | même modèle : 85,3 % (bloc B) contre 80,5 % (bloc D), 16 projets d'écart | Fragile | ± 5 points selon le pool : un écart de cette taille entre modèles ne veut rien dire. |
| Frontière α | test : 79,8 % (α = 1), 83,5 % (α = 0,5), 78,6 % (α = 0) ; impact 6,65 → 7,41 ; validation −13,0 points | Arbitrage | Plus de poids au comité : portefeuille mieux noté, mais coût en réussite en validation. |

**Conclusion.** Le ML prédit faiblement ; les agents justifient chaque décision avec des sources
vérifiées et extraient des facteurs de risque lisibles et prédictifs, mais n'ajoutent rien au texte
complet.
