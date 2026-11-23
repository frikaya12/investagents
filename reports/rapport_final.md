# InvestAgents : rapport final

*Sujet n°10 : « IA agentique pour la décision d'investissement multicritère : évaluation et
sélection de projets d'investissement ».*

Chaque chiffre de ce rapport vient d'un script du dépôt. Le rapport détaillé est indiqué entre
parenthèses. Les IC à 95 % sont obtenus par bootstrap (2 000 tirages). Classe positive des AUC :
projet **non satisfaisant** (MU/U/HU selon l'IEG). Split temporel sur l'exercice d'approbation :
entraînement ≤ 2007, validation 2008–2011, test 2012–2017. Le coffre-fort (projets évalués par
l'IEG en 2025 ou après) a été évalué une seule fois, à la fin. Tous les réglages ont été faits sur la
validation. Aucun choix (α, poids, modèle) n'a été fait en regardant le test.

## Réponse courte

- **Évaluation.** Sur le test 2012–2017, le ML prédit l'échec un peu mieux que le hasard (M1-texte :
  AUC-PR 0,297 contre 0,216).
  - **Sur le coffre-fort (projets récents, données neuves), M1 tombe au niveau du hasard.**
    M1-texte n'y est pas concluant (78 projets seulement).
  - Agents juges (notes M3/M4, sorties du débat M7) : pas d'information prédictive démontrée.
  - **Agents extracteurs (bloc D)** : leurs facteurs factuels portent seuls un signal faible
    mais réel (M5 − M0 significatif), mais rien de plus que le texte (M6 − M1-texte ≈ 0). Ils
    n'ont pas pu être évalués sur le coffre, faute de textes.
- **Multicritère.** Les agents apportent un **profil sur 5 critères** que le ML n'a pas. Mêlé à la
  probabilité de réussite (frontière α), ce profil donne des portefeuilles mieux notés par le
  comité. Au test, la perte de réussite n'est pas mesurable, mais elle l'est sur la validation.
- **Sélection.** Le sac à dos sous budget bat le hasard au test grâce aux probabilités du ML, pas
  grâce à la taille des projets.
- **Agentique.** Le comité est fiable : 600/600 projets, 99,8 % des citations retrouvées mot pour mot,
  0/50 projet reconnu. Il **explique** chaque projet avec des arguments sourcés. Les extracteurs
  (1 582 dossiers, 97,6 % de citations valides) transforment le dossier en facteurs lisibles. C'est
  leur apport réel ; ce n'est pas un meilleur classement que le texte seul.

## Métriques finales

| # | Partie | Métrique | Valeur [IC 95 %] | Référence | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | Données | Projets exploitables (dossier ex ante en texte, daté avant l'approbation) | 8 260 | seuil 300 | faisabilite.md |
| 2 | Évaluation ML | M1, AUC-PR, test (n = 1 294) | 0,247 [0,216 ; 0,286] | M0 0,217 | phase2_resultats.md |
| 3 | Évaluation ML | M1-texte figé, AUC-PR, test avec texte (n = 755) | 0,297 [0,245 ; 0,369] | M1 0,259 ; M0 0,216 | blocB_resultats.md |
| 4 | Évaluation ML | M1-texte − M1, AUC-PR (apparié) | +0,039 [0,003 ; 0,082] | 0 | blocB_resultats.md |
| 5 | Évaluation ML | **Coffre-fort : M1, AUC-PR / AUC-ROC (n = 700)** | 0,172 [0,144 ; 0,211] / 0,509 [0,457 ; 0,562] | M0 0,176 / 0,5 | coffre_fort.md |
| 6 | Évaluation ML | Coffre-fort : M1-texte − M1, AUC-PR (78 projets, descriptif) | +0,122 [0,002 ; 0,298] | 0 | coffre_fort.md |
| 7 | Évaluation agents | M3 (5 notes du Juge), AUC-PR, test (n = 300) | 0,200 [0,152 ; 0,272] | M0 0,213 | blocC_m3_m4.md |
| 8 | Évaluation agents | M4 − texte, AUC-PR | −0,031 [−0,088 ; 0,029] | 0 | blocC_m3_m4.md |
| 9 | Évaluation agents | M7 (texte + verdicts du critique, réécritures, TF-IDF des arguments Contre) − texte, AUC-PR | +0,027 [−0,059 ; 0,112] | 0 | blocC_m7.md |
| 10 | Multicritère | Frontière α, test, 25 % du budget : projets réussis α = 1 / 0,5 / 0 | 79,8 % / 83,5 % / 78,6 % (IC ± 8 pts) | hasard 76,7 % | frontiere_alpha.md |
| 11 | Multicritère | Idem, note d'impact moyenne du portefeuille (sur 10) α = 1 → 0 | 6,65 → 7,41 | hasard 6,49 | frontiere_alpha.md |
| 12 | Sélection | Portefeuille figé (Σ p × coût), 25 % du budget, test : projets satisfaisants | 85,3 % [79,3 ; 90,7] | hasard 77,2 % (p = 0,001) | blocB_resultats.md |
| 13 | Sélection | Contrôle M0 (taille seule), 25 % du budget | 76,4 % (p = 0,64) | hasard 77,2 % | blocB_resultats.md |
| 14 | Agentique | Projets notés, erreurs, temps médian | 600 / 600, 0, 89 s (T4), 0 € | — | blocC_comite.md |
| 15 | Agentique | Citations retrouvées mot pour mot dans le texte source | 5 986 / 6 000 = 99,8 % | — | blocC_comite.md |
| 16 | Agentique | Reconnaissance : projet « reconnu » ; note IEG proposée | 0 / 50 ; 0 / 50 | — | blocC_comite.md |
| 17 | Agentique | Dispersion des notes (écart-type) ; notes ≥ 9 | 0,65 à 1,26 ; ≤ 0,5 % | — | blocC_comite.md |
| 18 | Agentique | Bloc D : passage complet (1 582 dossiers), erreurs, citations valides, temps médian | 0 erreur ; 97,6 % ; 28 s | pilote 99,8 % | blocD_passage_complet.md |
| 19 | Évaluation agents | M5 (19 facteurs extraits, seuls) − M0, AUC-PR / AUC-ROC, test (n = 729) | +0,027 [0,003 ; 0,070] / +0,066 [0,017 ; 0,114] | 0 | blocD_m6.md |
| 20 | Évaluation agents | M6 (texte + facteurs) − M1-texte, AUC-PR | +0,003 [−0,039 ; 0,043] | 0 | blocD_m6.md |
| 21 | Sélection | Backtest 25 % du budget, test : M6 / M1-texte | 86,1 % (p < 0,001) / 80,5 % (p = 0,12) | hasard 77,7 % | blocD_m6.md |

## 1. Évaluation (ML + agents)

**ML.**
- M1 (gradient boosting sur variables ex ante) bat le taux de base au test, de peu (ligne 2). La
  région compte le plus.
- Le texte du dossier ajoute de l'information (lignes 3–4). Masquer les noms de pays coûte environ
  0,01 d'AUC-PR sur la validation : le signal du texte ne vient pas surtout de l'identité du pays.
- **Coffre-fort (données neuves) : M1 ne prédit plus rien** (ligne 5), y compris pour les projets
  approuvés ≤ 2017. Raisons possibles :
  - la dérive du taux d'échec (17,6 % contre 28 % à l'entraînement) ;
  - des projets plus récents que l'entraînement ;
  - un test 2012–2017 consulté pendant la phase 2.
- M1-texte n'a pu être mesuré que sur 78 projets du coffre (le serveur de documents a refusé les
  téléchargements, protocole amendé avant lecture). Il y fait mieux que M1 en AUC-PR (ligne 6),
  mais son AUC-ROC est compatible avec le hasard (0,543 [0,376 ; 0,724]) et il n'y a que 12 échecs :
  c'est descriptif.

**Agents.**
- Les 5 notes du Juge ne prédisent pas l'échec : AUC-ROC de 0,42 à 0,53 par critère sur la
  validation, et M3 au niveau du hasard sur le test (ligne 7). Ajoutées au texte, elles dégradent
  un peu les probabilités (ligne 8).
- Les sorties intermédiaires du débat (M7, ligne 9) donnent un écart positif au test, mais
  l'intervalle contient zéro et la validation annonçait l'inverse (0,345 contre 0,366 pour le texte
  seul).
- **Agents extracteurs (bloc D)** : 19 facteurs factuels lus dans le dossier, avec citations.
  - Seuls (M5), ils battent le taux de base (ligne 19), avec une AUC-ROC de 0,566, du même
    ordre que M1 sur les variables structurées.
  - Ajoutés au texte (M6), ils n'apportent rien (ligne 20) : ils sont lus dans le même document
    que celui qu'utilise M1-texte.
  - Associés à plus d'échecs : réinstallation de populations, nombre de cofinanceurs, deuxième
    phase, catégorie environnementale A, capacité faible de l'agence.

**Ce que les agents apportent ici :**
- les agents juges, aucun gain prédictif démontré ;
- les agents extracteurs, des variables lisibles et sourcées qui portent un signal faible, mais
  redondant avec le texte.

## 2. Multicritère (5 critères + frontière α)

- Le Juge note 5 critères inspirés des critères OCDE-CAD : pertinence, efficience économique,
  faisabilité, risque, impact et durabilité. Il note sur une grille à ancres (exemples de 2, 5
  et 8 par critère). Avec les ancres, il ne donne presque plus 9 ou 10 (ligne 17). Les critères
  sont peu corrélés entre eux (0,21 en moyenne) : ils mesurent des choses différentes.
- **Frontière α** : score = α × probabilité de réussite (ML) + (1 − α) × moyenne des 5 notes / 10.
  Les α sont fixés à l'avance, avec le même sac à dos et 25 % du budget.
  - De α = 1 à α = 0, la note d'impact du portefeuille monte de +0,75 point au test (+1,15 en
    validation). L'efficience et la faisabilité montent de +0,6 à +0,9 ; la pertinence ne bouge
    presque pas.
  - **Coût en réussite.** En validation : −13 points de projets réussis, le comité seul revenant
    au niveau du hasard. Au test : écarts dans le bruit (lignes 10–11).
  - Le gain en notes est mesuré avec les notes qui servent à choisir. Il est donc vrai par
    construction : il dit que le portefeuille suit mieux le jugement du comité, pas que les projets
    auront plus d'impact.
- **Aucun α n'est recommandé à partir du test.** Le choix de α est un choix du décideur (arbitrage
  réussite / critères). La démo permet de le faire sur les projets de la validation.
- **Ce que les agents apportent : une dimension de décision en plus** (le profil sur 5 critères),
  qui n'est pas validée par l'issue.

## 3. Sélection (sac à dos, backtest contre le hasard)

- Sac à dos exact OR-Tools CP-SAT, au plus 25 % du budget par pays, une sélection par cohorte
  annuelle. L'objectif Σ p × coût a été choisi sur la validation ; l'objectif initial Σ p ne battait
  pas le hasard.
- Au test, le portefeuille bat le hasard à 25 % et à 50 % du budget (ligne 12). Avec M0, il ne le
  bat pas (ligne 13) : le gain vient des probabilités du ML.
- **Bloc D** : avec les probabilités de M6, le portefeuille fait 86,1 % de projets réussis, contre
  80,5 % avec M1-texte et 77,7 % au hasard (ligne 21). **Cet écart n'établit pas que les facteurs
  sélectionnent mieux**, pour trois raisons :
  - l'écart n'est pas testé et les IC se recouvrent ;
  - M6 ne prédit pas mieux que M1-texte (ligne 20) ;
  - le backtest est instable : retirer 16 projets du pool fait passer M1-texte de 85,3 % à
    80,5 %.
- **Ce que les agents apportent à la sélection :** avec α < 1, ils déplacent le portefeuille vers
  des projets mieux notés sur les critères, sans gain de réussite démontré.

## 4. Agentique

- **Architecture (bloc C).** Graphe LangGraph :
  1. extraction des phrases clés numérotées ;
  2. débat Pour / Contre par critère, avec citations par numéros de phrases ;
  3. critique (fort / faible / non étayé) ;
  4. raffinement (1 tour) ;
  5. Juge à grille ancrée ;
  6. vérification des citations en Python.

  Les sorties sont des JSON validés par Pydantic. Prompts adaptés de DIALECTIC (MIT). Le modèle est
  Qwen2.5 7B, sur 2 GPU T4 gratuits de Kaggle.
- **Fiabilité.**
  - 600 projets sans erreur, avec 99,8 % des citations retrouvées mot pour mot (lignes 14–15).
  - Le critique juge 49 % des arguments faibles et 0,7 % non étayés ; 41 % des arguments sont
    réécrits au raffinement.
- **Anti-fuite.**
  - Documents datés strictement avant l'approbation ; noms et identifiants masqués ; pas d'accès
    web.
  - Test de reconnaissance (ligne 16) : 0/50 projets reconnus et 0/50 notes IEG proposées. Un seul
    nom approché (« FMR 2 »), reconstitué à partir des sigles du texte.
- **Agents extracteurs (bloc D).** Grille figée de 19 facteurs de risque factuels : composantes,
  agences, cofinanceurs, catégorie environnementale, note de risque, réforme à adopter, phase 2,
  fragilité, etc.
  - Un seul appel par dossier sur des phrases présélectionnées en Python dans le dossier complet,
    avec citations par numéros et « non trouvé » autorisé. Les nombres sont comptés en Python.
  - Pilote sur 30 projets, avec des critères fixés à l'avance : citations 99,8 %, 27 s,
    19/19 facteurs.
  - Passage complet sur 1 582 dossiers : 0 erreur, citations 97,6 %, pas nettement pire que le
    pilote selon des seuils fixés avant.
  - Exactitude vérifiable sur le pilote : 14/15 (catégorie environnementale) et 13/13 (note de
    risque). first_operation (1/3 juste) a été exclu avant le réglage.
- **Démo.** La fiche projet montre la probabilité de réussite (ML), les arguments Pour / Contre avec
  leurs phrases citées, le verdict du critique et les 5 notes justifiées. La page multicritère a un
  curseur α et les poids des critères.
- **Ce que les agents apportent : l'explication.** Chaque note est justifiée par des arguments
  sourcés et vérifiables. C'est ce que le ML ne fait pas.

## Limites

1. **Masquage incomplet.** Le masquage laisse les sigles et certaines variantes du nom du projet
   (exemple « FMR 2 »), ainsi qu'environ une grande ville par document. Le comité n'a pas été
   relancé (décision d'Aya).
2. **Un seul passage du comité**, avec un modèle 7B quantifié et 1 tour de raffinement. Ablations,
   stabilité sur 3 passages et M2 (prompt unique) non faits, faute de quota gratuit.
3. M4 et M7 sont des régressions logistiques apprises sur les 300 projets de validation. Le test du
   comité (300 projets) a servi deux fois (M3/M4, puis M7).
4. **Le test 2012–2017 a été consulté pendant la phase 2** ; le coffre-fort montre que les
   chiffres de test étaient optimistes pour M1.
5. **Coffre-fort.**
   - M1 y est au niveau du hasard sur les projets récents.
   - M1-texte n'a pu être mesuré que sur 78 projets (refus 403 du serveur), de façon
     descriptive et non concluante.
   - Les agents (blocs C et D) n'ont pas été évalués sur le coffre, faute de textes.
6. **Bloc D.**
   - Exactitude des facteurs mesurée sur le pilote seulement.
   - Pas de facteurs pour l'entraînement ≤ 2007 (quota Kaggle) : M5/M6 sont appris sur
     853 projets.
   - Le backtest est sensible à la composition du pool (±5 points).
7. Les indicateurs pays (WDI, WGI) sont les valeurs révisées actuelles, pas celles connues à
   l'époque.

## Attributions

- Données IEG, relevés IDA/IBRD et WGI : © Banque mondiale, licence CC BY 4.0. World Bank Projects
  API, Documents & Reports API et Indicators API.
- GeoNames (via geonamescache) : CC BY 4.0.
- DIALECTIC (github.com/pantageepapa/DIALECTIC), licence MIT : structure du débat et prompts adaptés,
  dans `src/investagents/debate/`.
- Qwen2.5-7B-Instruct (Ollama), exécuté sur les GPU gratuits de Kaggle.
- Code écrit avec l'aide de Claude Code (Anthropic), relu et validé par Aya.
