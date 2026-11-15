# Rapport de faisabilité — phase 1

*InvestAgents, 15 novembre 2026. Les chiffres viennent de `reports/faisabilite_chiffres.md`, généré
par `uv run python -m investagents.data.feasibility` à partir d'appels réels aux trois API.*

## Question

Avons-nous au moins **300 projets** notés par l'IEG qui ont un **document d'évaluation ex ante en
texte**, daté avant l'approbation du projet ? (seuil du cahier des charges)

## Décision : **GO**

Le seuil est dépassé de très loin, même avec la borne basse la plus prudente.

| Mesure | Projets exploitables | Base |
| --- | ---: | --- |
| Projets IEG notés (cible binaire connue) | 12 511 | jeu IEG complet |
| **Échantillon** : 28 / 50 avec dossier d'évaluation en texte (56 %) | **7 006** extrapolés | IC 95 % Wilson : 5 293 à 8 612 |
| Échantillon, avec repli sur un document ex ante court : 34 / 50 (68 %) | 8 507 extrapolés | IC 95 % : 6 780 à 9 914 |
| **Recensement complet** : dossier d'évaluation daté avant l'approbation, avec `txturl` | **8 260** | compte réel, tous projets notés |
| Recensement corrigé du taux de texte vide (29 / 30 dans l'échantillon) | ≈ 7 985 | estimation |
| Seuil du cahier des charges | 300 | |

Les deux méthodes concordent : le recensement (66 % des projets notés) tombe dans l'intervalle de
l'échantillon (42 % à 69 %), et les 28 projets « exploitables » de l'échantillon sont tous retrouvés
par le recensement.

**Classes.** Parmi les 8 260 projets du recensement : 6 050 satisfaisants (73 %) et **2 210 non
satisfaisants** (27 %). Il y a donc largement de quoi tirer les ~300 projets des agents en stratifiant
par note et par secteur, et assez de cas négatifs.

**Périodes.** La couverture est bonne à toutes les époques (62 % à 74 % des projets notés) :

| Exercice d'approbation | Projets notés | Avec dossier d'évaluation | Part |
| --- | ---: | ---: | ---: |
| ≤ 1997 | 6 259 | 3 852 | 62 % |
| 1998–2007 | 2 745 | 2 020 | 74 % |
| 2008–2015 | 2 351 | 1 541 | 66 % |
| ≥ 2016 | 1 156 | 847 | 73 % |

## Méthode

1. **IEG** (`ieg.py`) : 12 663 lignes via l'API FinancesOne (DS00053), 12 662 projets distincts
   (un doublon, P122764, on garde la dernière évaluation), 12 511 notés.
2. **Projects API** (`projects_api.py`) : 22 729 projets, dont la date d'approbation par le
   Conseil (`boardapprovaldate`).
3. **Documents API** (`documents_api.py`) : pour 50 projets notés tirés au hasard (graine 42),
   liste de **tous** leurs documents (1 084 lignes, 50 types différents).
4. **Filtre anti-fuite** (`leakage.py`, 100 % couvert par les tests) : type en liste blanche,
   anglais, date **strictement** antérieure à l'approbation.
5. **Texte** : téléchargement réel du `txturl` de chaque document gardé de l'échantillon ;
   exploitable si ≥ 10 000 caractères (seuil fixé avant de regarder les données).
6. **Recensement** : liste de tous les documents des 3 types « dossier d'évaluation » (15 142 lignes),
   même filtre, jointure avec les 12 511 projets notés.

### Types de documents retenus (valeurs exactes de `docty`, vérifiées par appels réels)

| Niveau | Type | Rôle | Nb dans l'API |
| --- | --- | --- | ---: |
| 1 | `Project Appraisal Document` | dossier d'évaluation (PAD), depuis ~1998 | 7 005 |
| 1 | `Staff Appraisal Report` | ancien nom du dossier d'évaluation (SAR) | 6 258 |
| 1 | `Program Document` | dossier des prêts à l'appui des politiques (DPF) | 1 608 |
| 2 | `Memorandum & Recommendation of the President`, `President's Report` | présentation au Conseil | — |
| 2 | `Project Information Document` / `Program Information Document` avec « Appraisal Stage » dans le titre | fiche d'information au stade de l'évaluation | — |

Réponse à la question ouverte du cahier des charges : **la valeur de `docty_exact` du dossier
d'évaluation est `Project Appraisal Document`**, complétée par `Staff Appraisal Report` pour les
projets anciens et `Program Document` pour les DPF. Le filtre par projet est `projectid=P…`.

## Problèmes rencontrés

1. **Dates d'approbation manquantes.** 1 803 projets notés n'ont pas de date d'approbation dans la
   Projects API (la fiche existe mais est presque vide, vérifié à la main sur P155969). Repli
   prudent : un document n'est gardé que s'il est daté avant le 1er juillet qui ouvre l'exercice
   d'approbation IEG. Cela rejette à tort des PAD datés quelques mois avant l'approbation
   (ex. P121231, P148183 dans l'échantillon).
2. **Dates des documents au mois près ou postérieures de quelques jours.** Les vieux SAR sont datés
   en fin de mois (SAR de P000803 daté du 1978-04-30, approbation le 1978-04-18). Et 889 dossiers
   d'évaluation sont datés après l'approbation, de **6 jours en médiane** : vraisemblablement la
   version finale publiée après le Conseil. La règle stricte les rejette. Si on tolérait 31 jours,
   on gagnerait 921 projets. **Non appliqué** : c'est une décision sur la règle anti-fuite, à
   prendre par toi.
3. **Fiches multi-projets.** `projectid` peut contenir `"P514478,P518183"` : géré (une ligne par
   projet).
4. **Traductions.** Certains dossiers existent en arabe, espagnol, etc. : seul l'anglais est gardé.
5. **Valeurs manquantes après Parquet** (`txturl` relu comme NaN) : bug trouvé et corrigé, testé.

## Ce qui reste incertain

- **Qualité du texte au-delà de la longueur.** Seulement 30 dossiers lus ; l'OCR des documents des
  années 1970–80 est lisible sur les extraits contrôlés, mais il y a des coquilles
  (« restriction6 »). La qualité sur l'ensemble du corpus n'est pas mesurée.
- **Le contenu « ex ante » est déduit du type et de la date**, pas vérifié en lisant chaque document.
  Les dossiers d'évaluation ne devraient pas contenir l'issue, mais un document mal typé dans l'API
  passerait le filtre. À contrôler sur un échantillon en phase 3.
- **Connaissance préalable du LLM.** Les PAD sont publics : le test de reconnaissance du cahier des
  charges reste indispensable. Seuls 550 projets exploitables ont été évalués par l'IEG en exercice
  ≥ 2025 : candidat pour le jeu de test « propre », à affiner selon la date limite du modèle choisi.
- **Variables IEG à ne pas utiliser en phase 2.** `quality_at_entry`, `quality_of_supervision`,
  `bank_performance`, `me_quality`, `evaluation_*` et `final_closing_fy` sont jugés **après** le
  projet : ce sont des fuites si on les met dans M1. Les montants de la Projects API sont les
  montants actuels (financements additionnels compris) : à vérifier avant usage.
- **Petit échantillon.** n = 50 donne un intervalle large (42 % à 69 %) ; le recensement le complète.

## Sources et licences

IEG World Bank Project Performance Ratings, © Banque mondiale, licence CC BY 4.0 (téléchargé le
2026-11-15, `as_of_date` 13-Nov-2026). World Bank Projects API et Documents & Reports API v3.
