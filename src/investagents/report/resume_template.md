# InvestAgents : résumé de présentation

**Le projet en bref.** Données : notes de résultat IEG (Banque mondiale) et dossiers d'évaluation
écrits avant l'approbation, {{n_exploitable}} projets. Anti-fuite : documents datés avant l'approbation,
variables post-projet interdites, noms masqués. Découpage temporel : entraînement ≤ 2007, validation
2008–2011, test 2012–2017, coffre-fort (évalués en 2025 ou après) ouvert une seule fois. Modèles : M0
(taux de base), M1 (gradient boosting), M1-texte (TF-IDF) ; M3, M4, M7 sur les sorties du comité
d'agents (Qwen2.5 7B via Ollama : Pour / Contre, critique, Juge sur 5 critères) ; M5, M6 sur 19
facteurs de risque extraits avec citations (M2 non fait). Sélection : sac à dos CP-SAT sous budget et
frontière α entre probabilité de réussite et notes du comité.

| Étape | Métrique (IC 95 %) | Verdict | En clair |
| --- | --- | --- | --- |
| Données | {{n_exploitable}} projets exploitables (seuil 300) | Bon | Assez de vrais projets pour apprendre et tester. |
| Anti-fuite | {{rec_recognized}} projet reconnu par le LLM ; colonnes post-projet bloquées par un test | Bon | Les agents ne trichent pas avec leur mémoire. |
| M0 · taux de base | AUC-PR {{m0_aucpr}} (test) | Référence | Le score du hasard, le plancher à battre. |
| M1 · gradient boosting | AUC-PR {{m1_aucpr}} ; AUC-ROC {{m1_aucroc}} | Faible | Les variables chiffrées battent à peine le hasard ; seule la région compte. |
| M1-texte · TF-IDF | AUC-PR {{m1t_aucpr}} contre {{base_test_text}} ; écart avec M1 {{m1t_minus_m1}} | Faible mais réel | Le meilleur modèle : le texte du dossier prédit un peu l'échec. |
| Coffre-fort · M1 | AUC-PR {{vault_m1_aucpr}} contre {{vault_m0_aucpr}} ; AUC-ROC {{vault_m1_aucroc}} | Négatif | Sur des projets neufs, M1 ne prédit plus rien : niveau du hasard. |
| Coffre-fort · M1-texte | écart avec M1 {{vault_m1t_minus_m1}} ({{vault_text_n}} projets) | Non concluant | Trop peu de textes téléchargés (refus du serveur) pour conclure. |
| Comité · fiabilité | {{com_judged}} projets, {{com_errors}} erreur ; citations exactes {{com_quotes_pct}} | Bon | Le comité tourne à l'échelle et cite vraiment le dossier. |
| M3 · notes du Juge | AUC-PR {{c_m3_aucpr}} contre {{c_base}} | Négatif | Les notes des juges ne prédisent pas l'échec. |
| M4 · texte + notes | écart avec le texte {{m4_minus_text}} | Négatif | Ajouter les notes au texte n'apporte rien. |
| M7 · texte + débat | écart avec le texte {{m7_minus_text}} | Non démontré | Zéro est dans l'intervalle ; la validation annonçait l'inverse. |
| Comité · qualité du débat | Pour et Contre citent les mêmes phrases dans {{ident_cit_pct}} des paires ({{ident_cit_n}} sur {{ident_cit_total}}) | Limite | Une citation exacte ne garantit pas un argument pertinent. |
| Extraction · fiabilité | {{d_full_n}} dossiers, {{d_full_err}} erreur ; citations valides {{d_full_cit}} | Bon | L'extracteur lit les dossiers à l'échelle avec des citations vérifiées. |
| M5 · 19 facteurs | écart avec M0 {{m5_minus_m0}} | Faible mais réel | Les facteurs extraits portent un vrai signal, petit. |
| M6 · texte + facteurs | écart avec le texte {{m6_minus_text}} | Négatif | Ce signal est déjà dans le texte complet. |
| Sélection CP-SAT | {{pf25}} de projets réussis (25 % du budget) contre {{pf25_random}} au hasard ({{pf25_p}}) | Bon | Le portefeuille choisi réussit mieux que le hasard. |
| Contrôle M0 | {{pf25_m0}} ({{pf25_m0_p}}) | Contrôle | Choisir sur la seule taille ne bat pas le hasard : le gain vient du ML. |
| Fragilité du backtest | même modèle : {{pf25_share}} (bloc B) contre {{bt_text}} (bloc D), {{bt_pool_gap}} projets d'écart | Fragile | ± 5 points selon le pool : un écart de cette taille entre modèles ne veut rien dire. |
| Frontière α | test : {{frt_1_0}} (α = 1), {{frt_0_5}} (α = 0,5), {{frt_0_0}} (α = 0) ; impact {{frt_1_0_impact}} → {{frt_0_0_impact}} ; validation {{frv_success_change}} | Arbitrage | Plus de poids au comité : portefeuille mieux noté, mais coût en réussite en validation. |

**AUC-ROC au test, IC à 95 %** (0,5 = hasard, quel que soit le taux d'échec). Chaque modèle est
mesuré sur son propre échantillon de test : on compare d'abord à 0,5, puis au texte seul sur le
même échantillon.

| Modèle | Échantillon de test | AUC-ROC [IC 95 %] | IC entièrement au-dessus de 0,5 ? |
| --- | --- | --- | --- |
| M0 · taux de base | test complet (n = {{n_test}}) | {{roc_m0}} | — (hasard par construction) |
| M1 · gradient boosting | test complet (n = {{n_test}}) | {{roc_m1}} | {{roc_m1_above}} |
| M1 · gradient boosting | projets avec texte (n = {{n_test_text}}) | {{roc_m1_text_subset}} | {{roc_m1_text_subset_above}} |
| M1-texte · TF-IDF | projets avec texte (n = {{n_test_text}}) | {{roc_m1t}} | {{roc_m1t_above}} |
| Texte seul (référence du comité) | test du comité (n = {{c_n_test}}) | {{roc_c_texte}} | {{roc_c_texte_above}} |
| M3 · 5 notes du Juge | test du comité (n = {{c_n_test}}) | {{roc_m3}} | {{roc_m3_above}} |
| M4 · texte + notes | test du comité (n = {{c_n_test}}) | {{roc_m4}} | {{roc_m4_above}} |
| M7 · texte + débat | test du comité (n = {{c_n_test}}) | {{roc_m7}} | {{roc_m7_above}} |
| Texte seul (référence du bloc D) | test du bloc D (n = {{d_n_test}}) | {{roc_d_texte}} | {{roc_d_texte_above}} |
| M5 · 19 facteurs | test du bloc D (n = {{d_n_test}}) | {{roc_m5}} | {{roc_m5_above}} |
| M6 · texte + facteurs | test du bloc D (n = {{d_n_test}}) | {{roc_m6}} | {{roc_m6_above}} |
| M1 · coffre-fort | projets évalués en 2025+ (n = {{vault_n}}) | {{roc_vault_m1}} | {{roc_vault_m1_above}} |

*Lecture : les AUC-ROC restent modestes (au mieux {{roc_max}}). M3 est sous 0,5 et son IC
contient 0,5 : les notes du Juge ne classent pas mieux que le hasard. Le texte seul du comité a un
IC qui touche 0,5, faute de projets ({{c_n_test}}). Sur le coffre-fort, M1 tombe à 0,5. Ce tableau ne dit
pas si un modèle bat un autre : pour cela, voir les écarts appariés.*

**Conclusion.** Le ML prédit faiblement ; les agents justifient chaque décision avec des sources
vérifiées et extraient des facteurs de risque lisibles et prédictifs, mais n'ajoutent rien au texte
complet.
