# InvestAgents : rapport final

*Projet de Machine Learning, sujet n°10 : « IA agentique pour la décision d'investissement
multicritère : évaluation et sélection de projets d'investissement ». Auteure : Aya (élève
ingénieure, data science / IA). Novembre 2026.*

*Ce document existe en trois versions identiques : `reports/rapport_final.md`, `docs/rapport.html`
(avec sommaire et glossaire cliquables) et `reports/rapport_final.pdf`. Les trois sont générées par
`uv run python -m investagents.report.build` à partir des fichiers de résultats du dépôt : aucun
chiffre n'est tapé à la main, et un test automatique vérifie que chaque chiffre correspond aux JSON
de résultats.*

## 0. Résumé en une page

> **En simple.** La Banque mondiale finance des milliers de projets ; environ un sur quatre est jugé
> en échec une fois terminé. Nous avons demandé à des agents d'IA de lire le dossier écrit *avant*
> le financement, de débattre du projet et de le noter, puis à un optimiseur de choisir quels
> projets financer avec un budget limité. Nous avons comparé tout cela à des modèles de machine
> learning classiques, sur des projets que personne n'avait regardés.

**La question.** Un comité d'agents d'IA qui lit les dossiers d'évaluation classe-t-il les projets
mieux qu'un modèle de machine learning classique ? Ses notes apportent-elles une information en
plus ?

**La méthode en 5 lignes.**

1. {{n_exploitable}} vrais projets de la Banque mondiale, avec la note finale de l'[[IEG]]
   (succès ou échec) comme vérité terrain.
2. Des règles anti-fuite strictes : seuls des documents datés *avant* l'approbation, noms masqués,
   [[découpage temporel]] (on apprend sur le passé, on teste sur l'avenir).
3. Des modèles de référence : [[gradient boosting]] sur des variables chiffrées, et
   [[TF-IDF]] + [[régression logistique]] sur le texte du dossier.
4. Deux familles d'[[agent|agents]] : des *juges* (débat Pour / Contre, critique, Juge sur 5
   critères) et des *extracteurs* (19 facteurs de risque factuels avec citations).
5. Un optimiseur ([[sac à dos (CP-SAT)]]) qui choisit un portefeuille sous budget, testé contre
   des portefeuilles tirés au hasard.

**Les 5 résultats clés.**

1. **Le ML prédit l'échec, mais faiblement.** Le meilleur modèle (texte du dossier) obtient une
   [[AUC-PR]] de {{m1t_aucpr}} au test, contre {{base_test_text}} pour le hasard. Sur des données
   neuves (le [[coffre-fort]]), le modèle à variables chiffrées tombe au niveau du hasard.
2. **Les agents juges n'ajoutent pas d'information prédictive.** Leurs 5 notes ne prédisent pas
   l'échec ({{c_m3_aucpr}} contre un taux de base de {{c_base}}).
3. **Les agents extracteurs produisent des facteurs lisibles et un peu prédictifs**
   (écart avec le hasard : {{m5_minus_m0}}), mais n'apportent rien de plus que le texte complet
   ({{m6_minus_text}}).
4. **Les agents justifient chaque décision avec des sources vérifiées** :
   {{com_quotes_pct}} des citations retrouvées mot pour mot, {{rec_recognized}} projet reconnu
   de mémoire.
5. **La sélection sous budget bat le hasard** au test ({{pf25_share}} de projets réussis contre
   {{pf25_random}}, {{pf25_p}}), grâce aux probabilités du ML et non à la taille des projets.

**Conclusion.** *Le ML prédit faiblement ; les agents justifient chaque décision avec des sources
vérifiées et extraient des facteurs de risque lisibles et prédictifs, mais n'ajoutent rien au texte
complet.* Le résultat principal est donc **négatif** pour la question de recherche, et
**positif** pour l'explication des décisions.

## 1. Contexte et question de recherche

> **En simple.** Choisir des projets à financer, c'est décider sous incertitude, avec plusieurs
> critères et un budget. Le sujet demande si des agents d'IA peuvent aider à *évaluer* les projets
> puis à les *sélectionner*. Nous avons pris de vrais projets dont on connaît la fin pour pouvoir
> vérifier.

**En technique.** Le sujet se décompose mot par mot :

| Mot de l'énoncé | Ce que nous avons construit |
| --- | --- |
| **IA agentique** | Un comité de [[LLM]] à rôles (avocat Pour, avocat Contre, critique, Juge) orchestré par [[LangGraph]], et des agents extracteurs de facteurs (sections 6 et 7). |
| **Décision d'investissement** | Choisir quels projets financer sous un budget fixé (section 8). |
| **Multicritère** | 5 critères inspirés des [[critères OCDE-CAD]] : pertinence, efficience économique, faisabilité, risque, impact et durabilité ; combinés avec la probabilité de réussite par la [[frontière α]] (section 8). |
| **Évaluation** | Prédire, avant l'approbation, si un projet sera en échec selon l'IEG (sections 5 à 7). |
| **Sélection** | Sac à dos exact sous budget avec plafond par pays, comparé au hasard (section 8). |

**Question de recherche.** Le comité d'agents classe-t-il les projets mieux qu'un gradient boosting
sur variables structurées ou qu'un modèle de texte simple ? Ses notes apportent-elles une
information en plus ? Vérité terrain : la note de résultat de l'IEG, binarisée en *succès*
(HS, S, MS) contre *échec* (MU, U, HU).

@@fig:pipeline@@

*Ce qu'il faut retenir : chaque étape alimente la suivante, et la vérité terrain (la note IEG)
n'est jamais montrée aux agents ni aux modèles avant l'évaluation.*

## 2. Données

> **En simple.** Pour chaque projet, nous avons la note finale de l'évaluateur indépendant de la
> Banque mondiale, et le dossier écrit avant le financement. Nous cherchons à prédire les échecs,
> qui sont minoritaires.

**En technique.**

- **Notes IEG** (licence CC BY 4.0) : note de résultat de chaque projet évalué. Cible
  `failure = 1` si MU, U ou HU (classe minoritaire).
- **Dossiers ex ante** ([[PAD]] ou *Staff Appraisal Report*) : World Bank Documents & Reports API,
  en anglais, texte d'au moins 10 000 caractères, daté strictement avant l'approbation.
- **Recensement (phase 1)** : {{n_exploitable}} projets notés ont un dossier exploitable
  (seuil go / no-go : 300), dont {{n_fail_exploitable}} en échec.
- **Variables structurées** (pour M1) : région, pratique et secteur, type d'instrument et
  d'accord, exercice d'approbation, montant à l'approbation (relevés IDA/IBRD, jamais le montant
  actuel), indicateurs pays (WDI). {{n_rated}} projets notés
  avec ces variables : {{n_train}} pour l'entraînement, {{n_valid}} pour la validation, {{n_test}}
  pour le test.
- **Avec texte** (pour M1-texte) : {{n_text_train}} / {{n_text_valid}} / {{n_text_test}}.
- **Dérive** : le taux d'échec baisse avec le temps, {{fail_train}} à l'entraînement, {{fail_valid}}
  en validation, {{fail_test}} au test, {{vault_fail}} sur le coffre-fort. C'est un
  [[décalage de domaine]], corrigé par recalibration sur la validation.

## 3. Anti-fuite

> **En simple.** Un modèle qui « voit l'avenir » a des scores trop beaux et inutiles. Nous avons
> donc interdit tout document écrit après l'approbation, masqué les noms des projets, et vérifié que
> le LLM ne reconnaissait pas les projets de mémoire.

**En technique.** La [[fuite de données]] est la première menace d'un tel projet. Quatre barrières :

1. **Dates.** Seuls les documents datés *strictement avant* l'approbation (filtre unique
   `src/investagents/data/leakage.py`, couvert par des tests). Jamais d'ICR, d'ICRR ou de PPAR
   (rapports de fin de projet). La tolérance de 31 jours n'existe que comme test de sensibilité.
2. **Colonnes interdites.** Les variables IEG post-projet sont refusées par
   `assert_no_forbidden` (`src/investagents/ml/features.py`) : `quality_at_entry`,
   `bank_performance`, `quality_of_supervision`, `me_quality`, `final_closing_fy` et toute colonne
   `evaluation_*`. Un test échoue si l'une d'elles entre dans les variables.
3. **Masquage.** Noms et identifiants de projets masqués côté agents ; noms de pays et de lieux
   masqués (spaCy + listes GeoNames) dans le texte montré aux agents. Pas d'accès au web pour les
   agents. Le texte des documents est traité comme une donnée, jamais comme une instruction.
4. **[[test de reconnaissance|Test de reconnaissance]].** Sur {{rec_n}} dossiers masqués, on
   demande au LLM s'il reconnaît le projet : {{rec_recognized}} projet reconnu, {{rec_rating}}
   note IEG proposée.

## 4. Protocole d'évaluation

> **En simple.** On règle les modèles sur une période, on les note une seule fois sur une période
> plus récente, et un dernier paquet de projets reste sous clé jusqu'à la fin. Chaque règle a été
> enregistrée dans git *avant* de voir les résultats : les identifiants de commits en sont la preuve.

**En technique.**

- **[[découpage temporel|Découpage temporel]]** sur l'exercice d'approbation : entraînement
  ≤ 2007, validation 2008–2011, test 2012–2017.
- **Validation pour régler** : tous les hyperparamètres, l'objectif de sélection et les choix de
  codage sont faits sur la validation, souvent par [[validation croisée]].
- **Test une seule fois** : chaque modèle est figé (fichier `*_frozen.json` commité), puis évalué
  une fois sur le test, avec [[intervalle de confiance bootstrap|IC bootstrap]] à 95 % et
  [[écart apparié|écarts appariés]].
- **Coffre-fort** : projets évalués par l'IEG en exercice ≥ 2025, ouverts une seule fois à la fin.

**Preuves dans l'historique git** (règle commitée → résultat commité) :

| Étape | Règle figée avant le résultat | Résultat |
| --- | --- | --- |
| Filtre anti-fuite | `7ac3f95` | — |
| Split temporel et coffre-fort | `ab4fd2d` | — |
| M1 et M1-texte (phase 2) | `c8974e6`, `9fbad94` | `cc0cac1` |
| Objectif de sélection | `eb73040` (choisi sur la validation) | `eb73040` (test une fois) |
| M1-texte du bloc B | `fe7a113` | `fe7a113` |
| M3 / M4 (notes du Juge) | `3e5f2de` | `024d6b5` |
| Coffre-fort | `ab9150e`, amendement `a572980` | `ead4cbe` |
| Grille des 19 facteurs | `ddd4cb1` | — |
| Pilote du bloc D | critères dans `d990604` | `2864354` |
| M5 / M6 | `3bf9e25` (codage), `106c3f0` (seuils), `a3ccb07` (réglage) | `43180e4` |

Deux lignes ont la règle et le résultat dans le même commit (objectif, bloc B) : le script
enchaîne le choix sur la validation puis l'évaluation unique, sans retour en arrière possible.

## 5. Modèles ML

> **En simple.** M0 répond toujours « taux moyen d'échec » : c'est le plancher. M1 utilise des
> chiffres sur le projet et le pays. M1-texte lit les mots du dossier. Une petite correction finale
> rend leurs probabilités honnêtes.

**En technique.**

- **M0** : prédit le [[taux de base]] pour tous les projets (AUC-PR = taux de base).
- **M1** : [[gradient boosting]] (`HistGradientBoostingClassifier`) sur variables structurées ex
  ante (jeu `{{m1_features}}`). AUC-PR {{m1_valid_aucpr}} en validation ; au test,
  {{m1_aucpr}} contre {{m0_aucpr}} pour M0 ; écart apparié {{m1_minus_m0}} ;
  [[test de permutation]] : {{m1_perm_p}}. [[AUC-ROC]] {{m1_aucroc}}.
- **M1-texte** : [[TF-IDF]] du dossier complet + [[régression logistique]] (C = 1, réglé sur la
  validation, AUC-PR {{m1t_valid_aucpr}}). Au test (projets avec texte, n = {{n_test_text}}) :
  {{m1t_aucpr}} contre {{m1_on_text_aucpr}} pour M1 sur les mêmes projets et {{base_test_text}}
  pour le hasard. Écart M1-texte − M1 : {{m1t_minus_m1}}. Fort
  [[sur-apprentissage]] (AUC-PR {{m1t_train_aucpr}} à l'entraînement), contenu par la
  régularisation.
- **Le pays revient par le texte.** Les mots les plus influents sont souvent des noms de pays. Sur
  la validation, masquer les lieux ne coûte qu'environ 0,01 d'AUC-PR : l'essentiel du signal ne
  vient pas de l'identité du pays.
- **[[calibration de Platt|Calibration de Platt]]** apprise sur la validation seulement :
  [[ECE]] {{ece_before}} → {{ece_after}}, [[Brier]] {{brier_before}} → {{brier_after}}. Elle
  corrige la dérive avant la sélection sous budget.

## 6. Agents juges (bloc C)

> **En simple.** Pour chaque projet, deux agents plaident Pour et Contre sur 5 critères en citant
> des phrases du dossier ; un critique juge leurs arguments ; ils se corrigent ; un Juge note.
> Python vérifie que chaque phrase citée existe. Le comité est fiable et transparent, mais ses notes
> ne prédisent pas l'échec.

**En technique.** Graphe [[LangGraph]], sorties JSON validées par Pydantic, prompts adaptés de
DIALECTIC (licence MIT), [[LLM]] Qwen2.5 7B sur 2 GPU T4 gratuits (Kaggle), 0 €.

@@fig:comite@@

*Ce qu'il faut retenir : chaque note du Juge s'appuie sur des arguments qui citent le dossier par
numéros de phrases, et Python vérifie les citations ; aucun chiffre ne vient du LLM.*

- **Fiabilité.** {{com_judged}} projets notés ({{c_n_valid}} de validation, {{c_n_test}} de test),
  {{com_errors}} erreur, {{com_seconds}} par projet. [[citation vérifiée|Citations vérifiées]] :
  {{com_quotes}} retrouvées mot pour mot ({{com_quotes_pct}}).
- **Débat utile.** Le critique juge {{critic_weak}} des arguments faibles et {{critic_unsupported}}
  non étayés ; {{args_rewritten}} des arguments sont réécrits au raffinement.
- **Notes dispersées.** Grille à ancres (exemples de notes 2, 5 et 8 par critère) : écart-type de
  {{score_std_range}} selon le critère, au plus {{share_ge9_max}} de notes ≥ 9. Corrélation
  moyenne entre critères : {{score_corr}} (les critères mesurent des choses différentes).
- **M3** (régression logistique sur les 5 notes, réglée sur la validation) : AUC-PR
  {{c_m3_aucpr}} au test, taux de base {{c_base}} ; écart M3 − M0 : {{m3_minus_m0}}. Sur la
  validation, l'AUC-ROC de chaque critère va de {{judge_valid_auc_range}}.
- **M4** (texte + 5 notes) : écart M4 − texte {{m4_minus_text}}.
- **M7** (texte + verdicts du critique + réécritures + TF-IDF des arguments Contre) : écart
  M7 − texte {{m7_minus_text}} ; la validation croisée annonçait l'inverse ({{m7_cv}} contre
  {{m7_cv_text}} pour le texte seul).

## 7. Agents extracteurs (bloc D)

> **En simple.** Un autre agent lit le dossier et répond à 19 questions factuelles (combien de
> composantes ? une réinstallation de populations est-elle prévue ? l'agence a-t-elle une capacité
> faible ?), chaque réponse avec une citation, « non trouvé » étant permis. Ces facteurs prédisent
> un peu l'échec, mais pas mieux que le texte complet.

**En technique.**

- **Grille figée** (commit `ddd4cb1`, empreinte SHA-256 testée) de 19 facteurs : composantes,
  agences d'exécution, cofinanceurs, première opération, deuxième phase, réforme à adopter,
  catégorie environnementale, note de risque globale, risque fiduciaire, valeurs de référence,
  leçons d'un projet en difficulté, fragilité / conflit, capacité faible, fonds de contrepartie,
  réinstallation, exécution décentralisée, projet pilote, risque politique, rentabilité
  économique chiffrée.
- **Un appel par dossier** sur des phrases présélectionnées en Python dans le dossier complet ;
  citations par numéros ; les nombres sont comptés en Python.
- **Pilote** ({{d_pilot_n}} projets, critères fixés avant) : citations valides {{d_pilot_cit}},
  {{d_pilot_s}} par projet, {{d_pilot_factors}} facteurs trouvés dans plus de 30 % des dossiers.
- **Contrôle d'exactitude** (pilote, facteurs vérifiables à la main) : catégorie
  environnementale {{d_acc_env}}, note de risque {{d_acc_risk}}. « Première opération » (1 juste
  sur 3) a été exclue avant tout réglage.
- **Passage complet** : {{d_full_n}} dossiers, {{d_full_err}} erreur, citations valides
  {{d_full_cit}}, {{d_full_s}} par projet ; « nettement pire que le pilote » selon des seuils fixés
  avant : {{d_full_much_worse}}. Table clé → projet vérifiée : {{d_mapping}} correspondances
  exactes.
- **M5** (19 facteurs seuls) et **M6** (logit de M1-texte + facteurs) : régressions logistiques
  réglées par validation croisée sur {{d_n_valid}} projets de validation (C = {{d_C}}). Validation
  croisée : M5 {{d_cv_m5}}, M6 {{d_cv_m6}}, texte seul {{d_cv_text}}.
- **Test** (n = {{d_n_test}}, taux de base {{d_base}}) : M1-texte {{d_texte_aucpr}}, M5
  {{d_m5_aucpr}}, M6 {{d_m6_aucpr}}. Écarts appariés : **M5 − M0 = {{m5_minus_m0}}** (AUC-ROC
  {{m5_minus_m0_roc}}) ; **M6 − M1-texte = {{m6_minus_text}}**.

@@fig:poids@@

*Ce qu'il faut retenir : les facteurs associés à plus d'échecs sont la réinstallation de
populations, le nombre de cofinanceurs, une deuxième phase, la catégorie environnementale A et une
agence à capacité faible ; les poids sont petits (forte régularisation) et se lisent comme des
tendances, pas comme des effets causaux.*

## 8. Décision multicritère et sélection

> **En simple.** Avec un budget limité, on choisit le panier de projets qui maximise les chances de
> réussite pondérées par leur coût. On peut aussi donner du poids aux notes du comité : un curseur
> α règle l'équilibre entre « chances de réussite » et « qualité selon le comité ».

**En technique.**

- **[[sac à dos (CP-SAT)|Sac à dos]]** exact (OR-Tools CP-SAT), une sélection par cohorte annuelle,
  au plus 25 % du budget par pays. **Objectif figé : maximiser Σ p × coût**, choisi sur la
  validation. L'objectif initial Σ p ne battait pas le hasard ({{pf25_initial_p}} de projets
  réussis à 25 % du budget, {{pf25_initial_p_p}}).
- **Backtest au test** (pool de {{pf_n_pool}} projets, 25 % du budget, {{pf25_n}} projets choisis) :
  {{pf25}} de projets réussis, contre {{pf25_random}} en moyenne pour des portefeuilles tirés au
  hasard ({{pf25_p}}). À 50 % du budget : {{pf50_share}} ({{pf50_p}}).
- **Contrôle M0** (taille des projets seule, même objectif) : {{pf25_m0}} ({{pf25_m0_p}}). Le gain
  vient donc des probabilités du ML.
- **Bloc D** (pool de {{bt_n_pool}} projets) : M6 {{bt_m6}} ({{bt_m6_p}}), M1-texte {{bt_text}}
  ({{bt_text_p}}), hasard {{bt_random}}. Cet écart M6 / M1-texte n'est pas interprétable (limite 3).

@@fig:backtest@@

*Ce qu'il faut retenir : le portefeuille M1-texte sort de la zone des 95 % de portefeuilles tirés
au hasard ({{pf25_random_band}}), le contrôle M0 y reste ; l'écart entre M6 et M1-texte au bloc D
est du même ordre que l'instabilité du backtest.*

- **[[frontière α|Frontière α]]** : score = α × probabilité de réussite + (1 − α) × moyenne des 5
  notes / 10 ; α ∈ {1 ; 0,75 ; 0,5 ; 0,25 ; 0} fixés à l'avance ; ~300 projets notés par période.
  - De α = 1 à α = 0, la note d'impact du portefeuille augmente de {{frt_impact_gain}} point au
    test ({{frv_impact_gain}} en validation).
  - Coût en réussite : {{frv_success_change}} en validation ({{frv_1_0}} → {{frv_0_0}}) ; au test,
    écarts dans le bruit : {{frt_1_0}} (α = 1), {{frt_0_5}} (α = 0,5), {{frt_0_0}} (α = 0), hasard
    {{frt_random}}.
  - Le gain en notes est vrai par construction (les notes servent à choisir) : il dit que le
    portefeuille suit mieux le jugement du comité, pas que les projets auront plus d'impact.
  - **Aucun α n'est recommandé à partir du test.** Le choix de α revient au décideur ; la démo le
    permet sur les projets de validation.

@@fig:frontiere@@

*Ce qu'il faut retenir : donner du poids au comité augmente la note d'impact du portefeuille ; en
validation, cela coûte des projets réussis, au test l'effet n'est pas mesurable.*

## 9. Métriques finales

> **En simple.** Toutes les mesures du projet, chacune avec sa fourchette d'incertitude, la
> référence à battre et ce qu'elle prouve vraiment.

**En technique.** IC à 95 % par bootstrap (2 000 tirages). AUC sur la classe « échec ».

| # | Métrique | Valeur [IC 95 %] | Référence | Ce que ça prouve |
| --- | --- | --- | --- | --- |
| 1 | Projets exploitables (dossier ex ante daté avant l'approbation) | {{n_exploitable}} | seuil 300 | Le projet est faisable sur de vraies données. |
| 2 | M1, AUC-PR, test (n = {{n_test}}) | {{m1_aucpr}} | M0 {{m0_aucpr}} | Les variables chiffrées prédisent un peu ({{m1_minus_m0}}). |
| 3 | M1-texte, AUC-PR, test (n = {{n_test_text}}) | {{m1t_aucpr}} | M1 {{m1_on_text_aucpr}} ; M0 {{base_test_text}} | Le texte du dossier est le meilleur signal… |
| 4 | M1-texte − M1, AUC-PR (apparié) | {{m1t_minus_m1}} | 0 | … et apporte plus que les variables chiffrées. |
| 5 | Coffre-fort : M1, AUC-PR (n = {{vault_n}}) | {{vault_m1_aucpr}} | M0 {{vault_m0_aucpr}} | Sur données neuves, M1 ne prédit plus ({{vault_m1_minus_m0}}). |
| 6 | Coffre-fort : M1-texte − M1 (n = {{vault_text_n}}, descriptif) | {{vault_m1t_minus_m1}} | 0 | Non concluant : {{vault_text_fail_n}} échecs seulement. |
| 7 | M3 (5 notes du Juge), AUC-PR, test (n = {{c_n_test}}) | {{c_m3_aucpr}} | M0 {{c_base}} | Les notes du Juge ne prédisent pas l'échec. |
| 8 | M4 − texte, AUC-PR | {{m4_minus_text}} | 0 | Les notes n'ajoutent rien au texte. |
| 9 | M7 − texte, AUC-PR | {{m7_minus_text}} | 0 | Le débat n'ajoute rien de démontré au texte. |
| 10 | M5 (19 facteurs) − M0, AUC-PR, test (n = {{d_n_test}}) | {{m5_minus_m0}} | 0 | Les facteurs extraits portent un signal réel, faible. |
| 11 | M6 (texte + facteurs) − M1-texte, AUC-PR | {{m6_minus_text}} | 0 | Ce signal est déjà dans le texte complet. |
| 12 | Portefeuille figé (Σ p × coût), 25 % du budget : projets réussis | {{pf25}} | hasard {{pf25_random}} ({{pf25_p}}) | La sélection sous budget bat le hasard. |
| 13 | Contrôle M0 (taille seule), 25 % du budget | {{pf25_m0}} | hasard {{pf25_random}} ({{pf25_m0_p}}) | Le gain vient des probabilités, pas de la taille. |
| 14 | Frontière α, test : projets réussis α = 1 / 0,5 / 0 | {{frt_1_0}} / {{frt_0_5}} / {{frt_0_0}} | hasard {{frt_random}} | Pas de coût en réussite mesurable au test. |
| 15 | Frontière α, test : note d'impact α = 1 → 0 | {{frt_1_0_impact}} → {{frt_0_0_impact}} | hasard {{frt_random_impact}} | Le portefeuille suit mieux le comité (vrai par construction). |
| 16 | Comité : projets notés, erreurs, temps médian | {{com_judged}}, {{com_errors}}, {{com_seconds}} | — | Le comité fonctionne à l'échelle, à coût nul. |
| 17 | Citations retrouvées mot pour mot | {{com_quotes}} ({{com_quotes_pct}}) | — | Les arguments sont sourcés et vérifiables. |
| 18 | Test de reconnaissance : reconnu ; note IEG proposée | {{rec_recognized}} ; {{rec_rating}} | — | Pas de mémorisation détectée. |
| 19 | Extracteurs : dossiers, erreurs, citations valides | {{d_full_n}}, {{d_full_err}}, {{d_full_cit}} | pilote {{d_pilot_cit}} | L'extraction tient à l'échelle. |
| 20 | Backtest bloc D, 25 % du budget : M6 / M1-texte | {{bt_m6}} ({{bt_m6_p}}) / {{bt_text}} ({{bt_text_p}}) | hasard {{bt_random}} | À ne pas interpréter : instabilité de ± 5 points (limite 3). |

@@fig:auc@@

*Ce qu'il faut retenir : tous les modèles sont proches de leur taux de base ; seul le texte du
dossier (M1-texte, et M6 qui le contient) s'en écarte nettement, et M1 retombe au taux de base sur
le coffre-fort.*

@@fig:ecarts@@

*Ce qu'il faut retenir : trois apports sont démontrés (M1 − M0, M1-texte − M1, M5 − M0) ; aucun
apport des agents *en plus du texte* ne l'est (M4, M6 et M7 ont un intervalle qui contient zéro).*

## 10. Ce que les agents apportent / n'apportent pas

> **En simple.** Les agents ne rendent pas la prédiction meilleure, mais ils rendent chaque décision
> explicable, vérifiable et discutable.

**En technique.**

| Partie | Ce que les agents apportent | Ce qu'ils n'apportent pas |
| --- | --- | --- |
| Évaluation | Des facteurs de risque lisibles, sourcés, au signal réel (M5 − M0 = {{m5_minus_m0}}). | Aucune information prédictive en plus du texte complet (M4, M6, M7 : IC contenant 0). Les notes du Juge ne prédisent pas l'échec. |
| Multicritère | Un profil sur 5 critères que le ML n'a pas ; une frontière α pour arbitrer. | Ces notes ne sont pas validées par l'issue IEG. |
| Sélection | Avec α < 1, des portefeuilles mieux notés sur les critères. | Aucun gain de réussite démontré. |
| Agentique | Des arguments Pour / Contre sourcés ({{com_quotes_pct}} de citations exactes), un critique, des facteurs vérifiés, un test de reconnaissance négatif. | Pas de vérification de la justesse de chaque argument au-delà de la citation. |

## 11. Limites

> **En simple.** Le signal est faible, certaines vérifications n'ont pu être faites qu'une fois ou
> sur peu de projets, et quelques choix techniques ont des défauts connus. Les voici tous.

**En technique.**

1. **Signal faible.** Les meilleurs modèles restent proches du taux de base ; les AUC-ROC sont
   autour de 0,6. Le ML ne « prédit » pas l'échec d'un projet donné : il le classe un peu mieux que
   le hasard en moyenne.
2. **Test consulté en phase 2.** Le test 2012–2017 a été regardé pendant la mise au point de M1 :
   le retrait de la variable « pays » a été décidé après avoir vu le test. M1 et M1-texte ont
   ensuite été figés et le test n'a plus servi à aucun choix. Le coffre-fort, lui, n'a jamais été
   regardé avant la fin : il montre que les chiffres de test de M1 étaient optimistes.
3. **Instabilité du backtest (± 5 points selon le pool).** Même modèle (M1-texte), même objectif,
   même budget : {{pf25_share}} au bloc B, {{bt_text}} au bloc D ; seule différence : le retrait
   de {{bt_pool_gap}} projets sans date d'approbation connue. Les écarts de backtest de l'ordre de
   5 points entre deux modèles ne doivent pas être interprétés ; seule la comparaison au hasard sur
   le même pool est fiable.
4. **Coffre-fort incomplet.** Le serveur de documents a refusé les téléchargements (403) : M1-texte
   n'a été mesuré que sur {{vault_text_n}} projets ({{vault_text_fail_n}} échecs), de façon
   descriptive (protocole amendé et commité avant lecture). Les agents n'ont pas été évalués sur le
   coffre-fort.
5. **Masquage des sigles.** Le masquage laisse les sigles et certaines variantes du nom du projet,
   et environ une grande ville par document. Le test de reconnaissance est négatif, mais le
   masquage n'est pas parfait.
6. **Un seul passage du LLM.** Un modèle 7B quantifié, un passage, un tour de raffinement :
   stabilité sur plusieurs passages, ablations et prompt unique (M2) non faits, faute de quota
   gratuit. Le test du comité ({{c_n_test}} projets) a servi deux fois (M3/M4, puis M7).
7. **Bloc D.** Exactitude des facteurs mesurée sur le pilote seulement ; pas de facteurs pour
   l'entraînement ≤ 2007 : M5/M6 sont appris sur {{d_n_valid}} projets.
8. **Indicateurs pays.** Les WDI sont les valeurs révisées actuelles, pas celles connues à l'époque.

## 12. Reproductibilité

> **En simple.** Tout le projet se relance avec quelques commandes ; les résultats de chaque étape
> sont dans le dépôt.

**En technique.** Python 3.12, `uv`. Chaque étape écrit ses résultats dans `experiments/` et son
rapport dans `reports/`.

```bash
uv sync                                                    # installer
uv run pytest && uv run ruff check .                       # tests (hors réseau) et lint
uv run python -m investagents.data.ieg                     # notes IEG
uv run python -m investagents.data.projects_api            # dates d'approbation
uv run python -m investagents.data.amounts                 # montants à l'approbation
uv run python -m investagents.data.country_indicators      # indicateurs pays
uv run python -m investagents.data.corpus                  # textes des dossiers (long)
uv run python -m investagents.ml.run_phase2 --stage select      # M1, M1-texte (validation)
uv run python -m investagents.ml.run_phase2 --stage evaluate    # test, une fois
uv run python -m investagents.ml.run_blocB --stage select|evaluate
uv run python -m investagents.selection.choose_objective --stage select|evaluate|control
uv run python -m investagents.committee_run.build_dataset  # comité : jeu pour Kaggle
uv run python -m investagents.committee_run.analyze        # après le passage Kaggle
uv run python -m investagents.ml.run_blocC_models --stage select|evaluate   # M3 / M4
uv run python -m investagents.ml.run_m7 --stage select|evaluate
uv run python -m investagents.selection.frontier --stage valid|test
uv run python -m investagents.extract.build_dataset        # extracteurs : jeu pour Kaggle
uv run python -m investagents.extract.analyze --run experiments/blocD/kaggle_full1
uv run python -m investagents.ml.run_m6 --stage choices|select|evaluate
uv run python -m investagents.ml.run_vault --stage evaluate
uv run python -m investagents.report.build                 # ce rapport (md, html, pdf)
uv run streamlit run src/investagents/app/streamlit_app.py # démo
```

Les passages LLM tournent sur Kaggle (GPU T4 gratuits) : voir `kaggle/blocC/` et `kaggle/blocD/`.

## 13. Attributions et usage de l'IA

> **En simple.** Nous avons utilisé des données publiques, un projet open source et des modèles
> d'IA ; chacun est cité avec sa licence.

**En technique.**

- **Données** : notes IEG, relevés IDA/IBRD, WDI — © Banque mondiale, licence CC BY 4.0 (World Bank
  Projects API, Documents & Reports API, Indicators API).
- **GeoNames** (via geonamescache) : CC BY 4.0, pour le masquage des lieux.
- **DIALECTIC** (github.com/pantageepapa/DIALECTIC), licence MIT : structure du débat et prompts
  adaptés, avec attribution dans `src/investagents/debate/`.
- **Qwen2.5-7B-Instruct** (Alibaba, licence Apache 2.0), exécuté avec Ollama sur les GPU gratuits de
  Kaggle : c'est le LLM des agents.
- **Claude Code** (Anthropic) : assistant de programmation pour écrire le code, les tests et les
  rapports ; chaque étape a été relue et validée par Aya, et chaque chiffre vient de l'exécution du
  code.
- Code du projet : licence MIT.

## 14. Glossaire

@@glossary@@
