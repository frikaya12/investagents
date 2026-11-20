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
| 8 | Objectif initial Σ p, part de projets satisfaisants (budget 10 / 25 / 50 %) | 75,8 / 76,0 / 77,5 % | hasard 76,5 / 77,2 / 77,7 % | **Pas mieux que le hasard** (p = 0,58 / 0,70 / 0,55) : il favorise les petits projets |
| 9 | **Objectif figé Σ p × coût** (choisi sur la validation), part de projets satisfaisants, budget 25 % | **85,3 %** [79,3 ; 90,7] | hasard 77,2 % [72,6 ; 82,1] ; base 78,0 % | Meilleur que le hasard (p = 0,001) |
| 10 | Idem, budget 50 % ; part du budget allant à des projets satisfaisants | 83,9 % [79,1 ; 88,3] (p = 0,001) ; budget 85,7 % [79,6 ; 91,1] | hasard 77,7 % ; budget 82,1 % | Meilleur que le hasard en nombre ; en montant, gain limite (p = 0,050). À 10 % : pas significatif (p = 0,13 et 0,22) |
| 11 | Citations valides du comité, citations par numéros de phrases (2 projets, 20 citations) | **100 %** [84 ; 100] | version précédente (citations recopiées) : 25 % [11 ; 47] | Les citations existent toutes dans le document ; leur pertinence reste à juger |
| 12 | Sorties JSON valides du comité au 1er essai | 8 / 8 appels (2 versions × 2 projets × 2 appels) | — | Pydantic + schéma Ollama : format fiable |
| 13 | Temps par projet (comité, qwen2.5:3b, CPU, sans GPU) | 4,6 et 5,0 min | — | ~5 min/projet : 300 projets ≈ 25 h de calcul local |
| 14 | Stabilité des notes (3 passages × 10 projets) | non mesurée | — | À faire |
| 15 | Taux de reconnaissance des projets par le LLM | non mesuré | — | À faire avant toute conclusion sur les agents |

**Objectif de sélection.** L'objectif initial (Σ p_i sous budget) récompense surtout le **nombre** de
projets : le solveur prend les moins chers et ne bat pas le hasard. Trois objectifs ont été comparés
**sur la validation** avec un critère fixé à l'avance (part du budget allant à des projets satisfaisants) :
Σ p (79,2 %), Σ p × coût (95,1 %), Σ p × coût avec p ≥ taux de base (95,1 %). Σ p × coût a été figé, puis
évalué **une seule fois** sur le test (lignes 9 et 10). Le test est nettement en dessous de la
validation (attendu : M1-texte et sa calibration ont été réglés sur la validation). Détail :
`reports/objectif_selection.md`.

**Contrôle : d'où vient le gain ?** Même objectif figé, sur le test, avec d'autres probabilités.
Avec des probabilités constantes (M0 : seule la taille compte), le portefeuille ne bat pas le hasard
(p = 0,64 à 25 % du budget). Avec M1 (sans texte), le gain est proche de celui de M1-texte (81,8 %
contre 85,3 % de projets satisfaisants à 25 % du budget, intervalles qui se recouvrent). Le gain vient
donc des probabilités du modèle, pas de la préférence pour les gros projets, et pas spécifiquement du
texte.
