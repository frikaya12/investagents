# Bloc B : points à signaler (écrit à la main, à lire d'abord)

1. **Aucune variante ne bat M1-texte sur la validation.** Le modèle figé est donc M1-texte
   (dossier complet, TF-IDF, phase 2). Sections clés seules : 0,304 ; embeddings MiniLM : 0,272 ;
   empilement M1 + texte : 0,344 (estimé par validation croisée, donc pas tout à fait comparable) ;
   M1-texte : 0,355. Résultat négatif pour les variantes : 8 000 caractères de sections clés
   contiennent moins de signal que le dossier entier (jusqu'à 150 000 caractères).
2. **Le test de M1-texte n'est pas une donnée neuve.** Ce modèle avait déjà été évalué sur le test
   en phase 2. L'évaluation « unique » du bloc B est neuve pour les variantes (tfidf_masked) et
   pour le backtest, pas pour le modèle figé.
3. **Écart apparent avec la phase 2 (0,297 contre 0,332 en AUC-PR test pour M1-texte) : ce n'est
   pas un bug.** En phase 2, le modèle final était réentraîné sur entraînement + validation. Ici,
   tous les modèles sont entraînés sur l'entraînement seul (≤ 2007), car la validation sert au
   choix et à la recalibration de Platt. Même cause pour M1 : 0,259 ici contre 0,247 en phase 2
   (autre sous-ensemble : les 755 projets avec texte).
4. **Embeddings : C = 0,01 est au bord de la grille.** Le meilleur réglage est la plus forte
   régularisation essayée. Le signal des embeddings est faible ; élargir la grille ne changerait
   probablement pas le classement (0,272 contre 0,355), mais ce n'est pas vérifié.
5. **Masquage des pays : le texte perd peu sans les noms de lieux.** Sur la validation, les sections
   masquées font 0,294 contre 0,304 sans masquage. Sur le test, tfidf_masked − M1-texte donne
   −0,045 [−0,094 ; −0,003], mais cela mélange deux effets : le masquage et le passage aux sections
   clés. L'effet propre du masquage se lit sur la validation : environ −0,01. L'essentiel du signal
   du texte ne vient donc pas des noms de pays (limite : environ 1 grande ville par document reste
   visible).
6. **Contrôle de la sélection (section 4).**
   - M0 (taille seule) ne bat jamais le hasard. À 50 % du budget, il fait même moins bien :
     75,1 % contre 77,7 %.
   - M1 et le modèle figé battent le hasard en part de projets satisfaisants à 25 % et à 50 %. Le
     gain vient des probabilités du modèle, pas de la taille des projets.
   - Le modèle figé fait mieux que M1 à 25 % et à 50 % (85,3 et 83,9 % contre 81,8 et 82,0 %), mais
     ces écarts ne sont pas testés et sont inférieurs à la largeur des intervalles.
   - En part du budget, seul le modèle figé à 50 % atteint p = 0,050 ; tout le reste n'est pas
     significatif.

# Bloc B : variantes de modèles de texte (généré automatiquement)

Entraînement ≤ 2007 ; choix sur la validation 2008–2011 ; test 2012–2017 calculé une
seule fois.
Mêmes projets pour tous les modèles (ceux qui ont un texte). Classe positive = échec.

## 1. Validation (869 projets, taux d'échec 22.8%) — sert au choix

| Candidat | AUC-PR validation | Réglage |
| --- | ---: | --- |
| m1_text_full | 0.355 | phase 2 (C = 1) |
| tfidf_sections | 0.304 | {'C': 1.0} |
| tfidf_masked | 0.294 | {'C': 0.3} |
| emb_sections | 0.272 | {'C': 0.01} |
| stack | 0.344 | {'text_model': 'm1_text_full', 'C': 1.0, 'cv': '5 plis dans la validation'} |
| m1 (référence, sans texte) | 0.331 | phase 2 |

**Modèle figé : `m1_text_full`.** `tfidf_masked` est un diagnostic
(non candidat) : masquage v2 = spaCy + listes fixes de pays et de capitales
(GeoNames, accents ignorés). Limite : environ 1 grande ville par document reste
visible (58 % des mentions de villes ≥ 100 000 hab. hors capitales ; mesure sur 300
projets d'entraînement, reports/masquage_lieux.md) ; l'information pays est réduite,
pas supprimée.
Embeddings : morceaux redécoupés pour tenir dans 256 jetons (aucune troncature).

le score de `stack` est estimé par validation croisée à 5 plis dans la validation.

## 2. Test (755 projets, taux d'échec 21.6%)

Probabilités recalibrées (Platt) sur la validation.

| Modèle | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier après | ECE avant → après |
| --- | --- | --- | ---: | --- |
| chosen | 0.297 [0.245 ; 0.369] | 0.598 [0.546 ; 0.647] | 0.1662 | 0.045 → 0.023 |
| tfidf_masked | 0.252 [0.211 ; 0.308] | 0.572 [0.525 ; 0.621] | 0.1687 | 0.062 → 0.018 |
| m1 | 0.259 [0.213 ; 0.320] | 0.573 [0.523 ; 0.621] | 0.1686 | 0.064 → 0.037 |
| m1_text | 0.297 [0.245 ; 0.369] | 0.598 [0.546 ; 0.647] | 0.1662 | 0.045 → 0.023 |

| Écart apparié | AUC-PR | AUC-ROC | Brier |
| --- | --- | --- | --- |
| chosen - m1 | 0.039 [0.003 ; 0.082] | 0.026 [-0.010 ; 0.063] | -0.002 [-0.006 ; 0.001] |
| chosen - m1_text | 0.000 [0.000 ; 0.000] | 0.000 [0.000 ; 0.000] | 0.000 [0.000 ; 0.000] |
| tfidf_masked - m1_text | -0.045 [-0.094 ; -0.003] | -0.026 [-0.072 ; 0.015] | 0.002 [-0.002 ; 0.007] |

## 3. Backtest de portefeuille avec le modèle figé (objectif figé du bloc A)

| Budget | % projets satisfaisants [IC 95 %] | hasard | p | % budget → satisfaisants [IC 95 %] | hasard | p | Projets |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| 10% | 80.5% [72.4% ; 88.5%] | 76.5% | 0.127 | 83.6% [73.0% ; 92.8%] | 79.4% | 0.215 | 87 |
| 25% | 85.3% [79.3% ; 90.7%] | 77.2% | 0.001 | 82.7% [73.0% ; 90.9%] | 80.3% | 0.262 | 150 |
| 50% | 83.9% [79.1% ; 88.3%] | 77.7% | 0.001 | 85.7% [79.6% ; 91.1%] | 82.1% | 0.050 | 273 |

## 4. Contrôle de la sélection : même objectif, probabilités de M0 et de M1

M0 = probabilité constante (seule la taille des projets compte) ; M1 = sans texte.

| Budget | Modèle | % projets satisfaisants | hasard | p | % budget → satisfaisants | hasard | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | figé | 80.5% | 76.5% | 0.127 | 83.6% | 79.4% | 0.215 |
| 10% | m0 | 74.2% | 76.5% | 0.749 | 74.7% | 79.4% | 0.826 |
| 10% | m1 | 82.3% | 76.5% | 0.044 | 83.6% | 79.4% | 0.212 |
| 25% | figé | 85.3% | 77.2% | 0.001 | 82.7% | 80.3% | 0.262 |
| 25% | m0 | 76.4% | 77.2% | 0.641 | 82.4% | 80.3% | 0.288 |
| 25% | m1 | 81.8% | 77.2% | 0.035 | 84.8% | 80.3% | 0.116 |
| 50% | figé | 83.9% | 77.7% | 0.001 | 85.7% | 82.1% | 0.050 |
| 50% | m0 | 75.1% | 77.7% | 0.966 | 78.4% | 82.1% | 0.948 |
| 50% | m1 | 82.0% | 77.7% | 0.001 | 84.8% | 82.1% | 0.112 |
