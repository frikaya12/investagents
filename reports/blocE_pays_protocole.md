# Bloc E : la part du pays dans ce qui prédit l'échec — protocole (écrit avant tout résultat)

*Écrit et commité AVANT de lancer le modèle sans noms de pays et avant de lire un seul poids ou
résultat de ce bloc. Toute modification après coup sera signalée comme un écart au protocole.*

## Question

Quand M1-texte prédit l'échec, est-ce grâce au **contenu du dossier** ou parce qu'il reconnaît
le **pays** (par son nom ou l'adjectif de nationalité) ?

## 1. Liste des noms de pays et adjectifs (figée)

- Construite par `src/investagents/ml/country_terms.py`, écrite dans
  `experiments/blocE/country_terms.json` (empreinte SHA-256 dans le fichier), **sans regarder
  aucun poids de modèle ni aucune issue**.
- Sources :
  1. la colonne « pays » des données (noms officiels IEG, par exemple « Federal Republic of
     Nigeria ») ;
  2. la liste standard mledoze/countries (base de REST Countries, licence ODbL) : nom courant et
     nom officiel en anglais, adjectifs de nationalité anglais (masculin et féminin, par exemple
     « Indonesian ») ;
  3. les anciens noms déjà fixés pour le masquage (« Zaire », « Upper Volta », « Burma »…).
- Une valeur de la colonne qui ne contient aucun nom ni adjectif de la liste est une **région**
  (« Africa », « Central Asia », « Western Balkans », « West Bank and Gaza »…) : elle n'est pas
  supprimée. On mesure la part du pays, pas celle de la région. La liste exacte des régions
  exclues est dans le JSON.
- Les variantes `altSpellings` de la liste standard ne sont pas utilisées (noms en langue locale
  et mots courts ambigus).
- Résultat : 720 expressions supprimées ; pour étiqueter le vocabulaire, 280 mots « pays » et
  219 mots « adjectif de pays », hors mots génériques fixés dans le code (« republic »,
  « south », « city », « africa »…).

## 2. Modèle sans noms de pays (une seule mesure, descriptive)

- **Suppression** : dans chaque texte, chaque expression de la liste est remplacée par un espace
  (mot entier, sans tenir compte de la casse ni des accents, la plus longue d'abord, même motif
  que le masquage du bloc B ; expressions de 3 caractères ou moins ignorées, comme au masquage).
- **Modèle identique** à M1-texte d'origine : même texte (150 000 premiers caractères du dossier),
  même TF-IDF, même régression logistique (C = 1, figé en phase 2), **aucun réglage**.
  Entraîné sur l'entraînement (≤ 2007, projets avec texte), comme les prédictions d'origine
  (`experiments/mvp/predictions_raw.parquet`).
- **Même calibration** : Platt apprise sur les mêmes projets de validation que le bloc B.
- **Contrôle de reproduction (avant la mesure)** : le même code SANS suppression doit redonner les
  prédictions d'origine (écart absolu maximal < 1e-9 sur la validation et le test). Sinon, arrêt.
- **Mesure unique** sur le test 2012–2017, sur **les mêmes 755 projets** que M1-texte au bloc B
  (`experiments/blocB/test_predictions.parquet`). Le script refuse de recalculer si le résultat
  existe déjà. **Le coffre-fort n'est pas touché.**

## 3. Chiffres rapportés

- AUC-PR et AUC-ROC avec IC à 95 % (bootstrap, 2 000 tirages, graine 41) pour M1-texte d'origine
  (calibré, bloc B) et M1-texte sans pays ; taux de base.
- Écart apparié « sans pays − d'origine » (mêmes tirages, graine 42), AUC-PR et AUC-ROC.
- Part des textes supprimée (nombre moyen d'expressions et part des caractères), à titre
  descriptif.

**Règle de conclusion (fixée maintenant).** Soit Δ l'écart d'AUC-PR (sans pays − d'origine) et
r = (AUC-PR d'origine − AUC-PR sans pays) / (AUC-PR d'origine − taux de base), la part du gain
au-dessus du hasard perdue sans les noms de pays :

- r ≥ 0,5 → « le signal vient **surtout du pays** » ;
- sinon, si r ≥ 0,2 ou si l'IC de Δ est entièrement sous 0 → « le signal vient **en partie du
  pays** » ;
- sinon → « le signal vient **surtout du contenu du dossier** ».

## 4. Mots les plus lourds du modèle d'origine

- Poids de la régression logistique du modèle d'origine (celui du contrôle de reproduction,
  entraîné sur ≤ 2007). Poids positif = vers l'échec, négatif = vers la réussite.
- Les 20 mots les plus lourds de chaque côté, avec l'étiquette « pays » ou « adjectif de pays »
  selon la liste figée.
- Part des mots « pays » ou « adjectif de pays » parmi les 100 mots de plus grand poids absolu.

## 5. Facteurs extraits (bloc D) les plus liés à l'échec

- Données : les **853 projets de validation** du bloc D (là où M5 a été appris). Le test n'est pas
  relu pour ce point.
- Pour chacun des 19 facteurs, « présent » contre « absent » (réponse vérifiée seulement, comme M5) :
  - oui / non : présent = « oui » ;
  - catégorie environnementale : présent = « A » ;
  - note de risque globale et risque fiduciaire : présent = « élevé » ou « substantiel » ;
  - listes (composantes, agences, cofinanceurs) : présent = nombre strictement supérieur à la
    médiane des projets où le facteur est trouvé ;
  - « non trouvé » ou réponse non vérifiée = absent.
- Effet = taux d'échec si présent − taux d'échec si absent, en points ; IC à 95 % par bootstrap
  (2 000 tirages, graine 43). Classement par effet. Le poids du facteur dans M5 (JSON du bloc D)
  est rappelé à côté.
- Descriptif : ces effets ne sont pas causaux et ne sont pas ajustés les uns sur les autres.

## Sorties

`experiments/blocE/results.json` (tous les chiffres), rapport `reports/blocE_pays.md` (chiffres lus
par `report/metrics.py`), nouvelle section du rapport final et ligne du résumé, quel que soit le
résultat.

Lancer : `uv run python -m investagents.ml.run_blocE --stage evaluate` (une seule fois).
