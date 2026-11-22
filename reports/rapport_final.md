# InvestAgents : rapport final (v1.0)

*Sujet n°10 : « IA agentique pour la décision d'investissement multicritère : évaluation et
sélection de projets d'investissement ».*

Chaque chiffre de ce rapport vient d'un script du dépôt. Le rapport détaillé est indiqué entre
parenthèses. Les IC à 95 % sont obtenus par bootstrap (2 000 tirages). Classe positive des AUC :
projet **non satisfaisant** (MU/U/HU selon l'IEG). Split temporel sur l'exercice d'approbation :
entraînement ≤ 2007, validation 2008–2011, test 2012–2017.

**État du coffre-fort** (projets évalués par l'IEG en 2025 ou après) : protocole figé
(`experiments/vault/protocol.json`) ; textes en cours de téléchargement. Le chiffre du coffre-fort sera
ajouté dès qu'il aura été calculé, une seule fois (`reports/coffre_fort.md`).

## Question de recherche, réponse courte

Le comité d'agents (Qwen2.5 7B : débat Pour / Contre → critique → raffinement → Juge à grille ancrée)
**ne prédit pas mieux l'échec** qu'un modèle de texte simple (TF-IDF + régression logistique), et ses
5 notes n'ajoutent **pas d'information prédictive** mesurable. C'est un résultat négatif, mesuré une
seule fois sur le test. En revanche, le ML (variables structurées et texte) bat le taux de base, et
la sélection sous budget qui l'utilise bat le hasard.

## Métriques finales (test 2012–2017, chaque modèle évalué une fois après gel sur la validation)

| # | Métrique | Valeur [IC 95 %] | Référence | Source |
| --- | --- | --- | --- | --- |
| 1 | Projets exploitables (dossier ex ante en texte, daté avant l'approbation) | 8 260 | seuil 300 | faisabilite.md |
| 2 | M1 (gradient boosting, variables ex ante), AUC-PR, test complet (n = 1 294) | 0,247 [0,216 ; 0,286] | M0 0,217 | phase2_resultats.md |
| 3 | M1 − M0, AUC-PR (apparié) | +0,030 [0,009 ; 0,060] | 0 | phase2_resultats.md |
| 4 | M1-texte figé (entraîné ≤ 2007, Platt sur la validation), AUC-PR, test avec texte (n = 755) | 0,297 [0,245 ; 0,369] | M1 0,259 ; M0 0,216 | blocB_resultats.md |
| 5 | M1-texte, AUC-ROC (n = 755) | 0,598 [0,546 ; 0,647] | 0,5 | blocB_resultats.md |
| 6 | M1-texte − M1, AUC-PR (apparié) | +0,039 [0,003 ; 0,082] | 0 | blocB_resultats.md |
| 7 | M1-texte, ECE avant → après recalibration ; Brier | 0,045 → 0,023 ; 0,1662 | M0 0,173 | blocB_resultats.md |
| 8 | Texte masqué (sections, pays masqués) − M1-texte, AUC-PR | −0,045 [−0,094 ; −0,003] | 0 | blocB_resultats.md |
| 9 | Portefeuille (Σ p × coût, ≤ 25 % par pays), 25 % du budget : part de projets satisfaisants | 85,3 % [79,3 ; 90,7] | hasard 77,2 % (p = 0,001) | blocB_resultats.md |
| 10 | Idem à 50 % du budget ; part du budget allant à des projets satisfaisants | 83,9 % (p = 0,001) ; 85,7 % (p = 0,050) | hasard 77,7 % ; 82,1 % | blocB_resultats.md |
| 11 | Contrôle : même sélection avec M0 (taille seule), 25 % du budget | 76,4 % (p = 0,64) | hasard 77,2 % | blocB_resultats.md |
| 12 | Comité : projets notés, erreurs, temps par projet | 600 / 600, 0 erreur, médiane 89 s (T4), 0 € | — | blocC_comite.md |
| 13 | Citations vérifiées mot pour mot dans le texte source | 5 986 / 6 000 = 99,8 % | — | blocC_comite.md |
| 14 | Test de reconnaissance : projet « reconnu » ; note IEG proposée | 0 / 50 ; 0 / 50 | — | blocC_comite.md |
| 15 | Dispersion des notes du Juge (écart-type par critère) ; part des notes ≥ 9 | 0,65 à 1,26 ; ≤ 0,5 % | Juge 3B sans ancres : 10/10 fréquent | blocC_comite.md |
| 16 | M3 (régression logistique sur les 5 notes), AUC-PR, test (n = 300) | 0,200 [0,152 ; 0,272] | M0 0,213 | blocC_m3_m4.md |
| 17 | M4 (texte + 5 notes), AUC-PR, test (n = 300) | 0,263 [0,193 ; 0,361] | texte seul 0,294 [0,214 ; 0,396] | blocC_m3_m4.md |
| 18 | M4 − texte : AUC-PR ; Brier | −0,031 [−0,088 ; 0,029] ; +0,007 [0,000 ; 0,015] | 0 | blocC_m3_m4.md |
| 19 | M3 − M0, AUC-PR | −0,013 [−0,037 ; 0,046] | 0 | blocC_m3_m4.md |

## 1. Évaluation (ML + agents)

- **ML.** Les variables ex ante portent un signal faible mais réel (lignes 2–3). Le pays et la région
  comptent le plus. Le texte du dossier d'évaluation ajoute de l'information (ligne 6). Masquer les
  noms de pays coûte environ 0,01 d'AUC-PR sur la validation : le signal du texte ne se réduit donc
  pas à l'identité du pays.
- **Agents.** Les notes du comité ne prédisent pas l'échec : AUC-ROC de 0,42 à 0,53 par critère sur la
  validation, et M3 au niveau du hasard sur le test. Ajoutées au texte, elles dégradent un peu les
  probabilités (M4, ligne 18).
- **Ce que les agents apportent ici : rien de mesurable en prédiction.**

## 2. Multicritère

- Le Juge note 5 critères (pertinence, efficience économique, faisabilité, risque, impact et
  durabilité) sur une grille à ancres (exemples de 2, 5 et 8 par critère). Avec les ancres, le Juge
  ne donne presque plus 9 ou 10 (ligne 15). Les critères sont peu corrélés entre eux (corrélation
  moyenne 0,21) : ils mesurent des choses différentes.
- La démo propose une **sélection multicritère** : le décideur choisit le poids de chaque critère. Ces
  notes servent à classer selon ses préférences, **pas à prédire l'échec** (AUC sur la validation
  ≈ 0,5).

## 3. Sélection (sac à dos, backtest contre le hasard)

- Sac à dos exact OR-Tools CP-SAT, au plus 25 % du budget par pays, une sélection par cohorte annuelle.
  L'objectif Σ p × coût a été choisi sur la validation ; l'objectif initial Σ p ne battait pas le
  hasard.
- Sur le test, le portefeuille bat le hasard à 25 % et à 50 % du budget (lignes 9–10). Avec M0, il ne
  le bat pas (ligne 11) : le gain vient des probabilités du modèle, pas de la taille des projets.

## 4. Agentique

- **Architecture.** Graphe LangGraph : extraction des phrases clés numérotées → débat Pour / Contre
  par critère, avec citations par numéros de phrases → critique (fort / faible / non étayé) →
  raffinement (1 tour) → Juge à grille ancrée → vérification des citations en Python. Les sorties
  sont des JSON validés par Pydantic. Prompts adaptés de DIALECTIC (MIT).
- **Fiabilité.** 600 projets sans erreur, 99,8 % des citations retrouvées mot pour mot. Le critique
  juge 49 % des arguments faibles et 0,7 % non étayés ; 41 % des arguments sont réécrits au
  raffinement.
- **Anti-fuite.** Documents datés strictement avant l'approbation ; noms et identifiants masqués ;
  pas d'accès web. Test de reconnaissance : 0/50 projets reconnus, 0/50 notes IEG proposées.
- **Ce que les agents apportent : une analyse lisible et sourcée** de chaque dossier (arguments
  cités, profil sur 5 critères), **pas un meilleur classement**.

## Limites

1. **Masquage incomplet.** Le masquage laisse les sigles et certaines variantes du nom du projet
   (exemple « FMR 2 »), ainsi qu'environ une grande ville par document. Le comité n'a pas été
   relancé (décision d'Aya).
2. **Un seul passage du comité**, avec un modèle 7B quantifié et 1 tour de raffinement. Ablations,
   stabilité sur 3 passages et M2 (prompt unique) non faits, faute de quota gratuit.
3. M4 est un empilement logistique, pas un gradient boosting : avec 300 projets d'apprentissage, un
   gradient boosting sur-apprendrait.
4. **Le test 2012–2017 a été consulté pendant la phase 2.** Les chiffres de test sont donc
   indicatifs ; le chiffre neuf est celui du coffre-fort.
5. Les indicateurs pays (WDI, WGI) sont les valeurs révisées actuelles, pas celles connues à
   l'époque.
6. Les IC du comité sont larges (n = 300, 64 échecs) : un petit apport (moins de 0,05 d'AUC-PR) ne
   peut pas être exclu.

## Attributions

- Données IEG, relevés IDA/IBRD et WGI : © Banque mondiale, licence CC BY 4.0. World Bank Projects
  API, Documents & Reports API et Indicators API.
- GeoNames (via geonamescache) : CC BY 4.0.
- DIALECTIC (github.com/pantageepapa/DIALECTIC), licence MIT : structure du débat et prompts adaptés,
  dans `src/investagents/debate/`.
- Qwen2.5-7B-Instruct (Ollama), exécuté sur les GPU gratuits de Kaggle.
- Code écrit avec l'aide de Claude Code (Anthropic), relu et validé par Aya.
