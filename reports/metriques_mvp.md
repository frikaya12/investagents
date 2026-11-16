# Métriques du MVP

Chaque chiffre vient d'un script du dépôt. IC 95 % : bootstrap (2 000 tirages) pour les AUC, Wilson
pour les proportions, centiles 2,5 / 97,5 % pour le hasard (1 000 portefeuilles).

| # | Métrique | Valeur [IC 95 %] | Référence | Ce que ça prouve |
| --- | --- | --- | --- | --- |
| 1 | Projets exploitables (dossier ex ante en texte, daté avant approbation) | 8 260 (recensement) | seuil 300 | La faisabilité est largement acquise |
| 2 | AUC-PR M1, test complet (n = 1 294) | 0,247 [0,216 ; 0,286] | hasard 0,217 | Les variables ex ante portent un signal faible mais réel (p = 0,016) |
| 3 | AUC-ROC M1, test complet | 0,567 [0,529 ; 0,603] | 0,5 | Idem |
| 4 | AUC-PR M1-texte, test avec texte (n = 755) | 0,332 [0,272 ; 0,409] | M1 0,278 ; hasard 0,216 | Le texte du dossier d'évaluation apporte de l'information |
| 5 | Écart M1-texte − M1 (AUC-PR, apparié) | +0,054 [0,006 ; 0,112] | 0 | Gain du texte significatif en AUC-PR (pas en AUC-ROC : +0,038 [−0,002 ; 0,074]) |
| 6 | ECE M1-texte, test, avant → après recalibration (Platt, validation) | 0,045 → 0,023 | 0 = parfait | La recalibration sur la validation corrige la dérive du taux d'échec |
| 7 | Brier M1-texte, test, avant → après | 0,168 → 0,166 | M0 0,173 | Probabilités un peu meilleures que le taux de base |
| 8 | Part de projets satisfaisants dans le portefeuille, budget 10 % (M1-texte) | 75,8 % [70,9 ; 80,2] | hasard 76,4 % [69,4 ; 82,7] ; M1 74,6 % ; base 78,0 % | **Pas mieux que le hasard** (p = 0,57) |
| 9 | Idem, budget 25 % | 76,0 % [72,0 ; 79,6] | hasard 77,3 % ; M1 76,0 % ; base 78,0 % | Pas mieux que le hasard (p = 0,71) |
| 10 | Idem, budget 50 % | 77,5 % [74,1 ; 80,7] | hasard 77,7 % ; M1 77,4 % ; base 78,0 % | Pas mieux que le hasard (p = 0,54) |
| 11 | Citations du comité retrouvées dans le document (2 projets, 20 citations) | 25 % [11 ; 47] | cible ≥ 98 % | Le modèle 3B **paraphrase** : citations pas encore fiables |
| 12 | Sorties JSON valides du comité au 1er essai | 4 / 4 appels | — | Pydantic + schéma Ollama : format fiable |
| 13 | Temps par projet (comité, qwen2.5:3b, CPU sur batterie) | 6,1 et 8,2 min | — | ~7 min/projet : 300 projets ≈ 35 h de calcul local |
| 14 | Stabilité des notes (3 passages × 10 projets) | non mesurée | — | À faire |
| 15 | Taux de reconnaissance des projets par le LLM | non mesuré | — | À faire avant toute conclusion sur les agents |

**Pourquoi le portefeuille ne bat pas le hasard.** L'objectif du cahier des charges (maximiser
Σ p_i sous budget) récompense surtout le **nombre** de projets financés : le solveur prend les projets
les moins chers, et M1 comme M1-texte retiennent exactement le même nombre de projets (327, 488, 619).
Comme les probabilités calibrées varient peu d'un projet à l'autre, elles pèsent peu dans le choix.
L'objectif n'a **pas** été modifié après avoir vu le test (ce serait régler sur le test) : une
alternative (par exemple Σ p_i · c_i, ou un seuil de probabilité minimal) sera fixée à l'avance et
évaluée une seule fois sur le coffre-fort.
