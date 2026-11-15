# Phase 1 expliquée simplement : chaque fichier, ce qu'il fait, pourquoi

## Outillage du dépôt

| Fichier | Ce qu'il fait | Pourquoi |
| --- | --- | --- |
| `pyproject.toml` | Décrit le paquet `investagents` : version de Python (≥ 3.12), dépendances (requests, pandas, pyarrow, tabulate), outils de dev (pytest, pytest-cov, ruff) et leur configuration. | Un seul fichier de configuration standard. `uv sync` recrée exactement le même environnement. |
| `uv.lock` | Versions exactes de toutes les dépendances. | Reproductibilité : le jury ou la CI installent les mêmes versions que toi. |
| `.python-version` | Fixe Python 3.12 pour uv. | Ton PC a Python 3.11 ; uv a installé 3.12 à part, sans toucher au reste. |
| `.gitignore` | Exclut `.venv/`, caches, `data/raw/*`, `data/processed/*`, `.env`. | Les données sont volumineuses et re-téléchargeables ; les secrets ne doivent jamais être versionnés. |
| `.github/workflows/ci.yml` | À chaque push : installe, lance Ruff puis pytest. | « CI verte » exigée par le cahier des charges. |
| `README.md` | Présentation courte, commandes, structure, licences. | Point d'entrée du dépôt. |
| `src/investagents/*/__init__.py` | Dossiers vides de la structure du cahier des charges (agents, debate, graph, ml...). | Préparer l'architecture sans coder les phases suivantes. |

**Tests marqués `network`** : par défaut, `uv run pytest` n'appelle pas Internet (tests rapides,
CI stable). `uv run pytest -m network` vérifie que les vraies API répondent toujours au format
attendu.

## Code de données (`src/investagents/data/`)

### `http.py` — client HTTP commun
Une fonction `get_json(url, params)` : fait la requête et renvoie le JSON. Si le serveur répond 429
(« trop de requêtes », vu en vrai sur l'API du catalogue) ou une erreur 5xx, elle attend 5 s, 10 s,
20 s... et réessaie. Une erreur 404 n'est pas réessayée (elle ne se corrigera pas seule).
*Tests :* `tests/test_http.py` simule des réponses (succès, 429 puis succès, coupure réseau, abandon).

### `ieg.py` — la vérité terrain
- `fetch_ieg()` télécharge les 12 663 lignes IEG par pages de 1 000 et vérifie qu'on a bien reçu le
  nombre annoncé par l'API.
- `clean_ieg()` : convertit les types, crée `outcome_rank` (1 = HU ... 6 = HS) et la cible
  `satisfactory` (1 = MS/S/HS, 0 = MU/U/HU, vide = « Not Rated »). Un libellé inconnu provoque une
  erreur volontaire : on veut savoir si l'IEG change ses libellés. Le doublon P122764 est résolu en
  gardant l'évaluation la plus récente (règle de l'IEG).
- `summarize()` : comptes par note et par année d'évaluation.
*Pourquoi* : sans cible propre, aucun modèle n'est évaluable.

### `projects_api.py` — la date d'approbation
Télécharge les 22 729 projets de la Projects API et extrait la **date d'approbation par le
Conseil**, indispensable au filtre anti-fuite, plus montant, instrument, région, pays.
Les formats ont été relevés sur une vraie réponse (« 50,000,000 » en texte, dates au format
« 12/31/2024 12:00:00 AM ») et sont testés.

### `documents_api.py` — les documents
- `fetch_documents()` / `fetch_project_documents()` : tous les documents d'un projet (paramètre
  `projectid`, vérifié) ou d'un type (paramètre `docty_exact`, vérifié), page par page.
- `parse_documents()` : une ligne par document et par projet (un document peut concerner deux
  projets : « P514478,P518183 »).
- `sample_rated_projects()` : tirage de 50 projets notés avec une graine fixe (42) → reproductible.
- `download_text()` : télécharge le texte d'un document, le met en cache dans `data/raw/texts/`, et
  renvoie sa longueur (0 si absent).

### `leakage.py` — le filtre anti-fuite (le fichier le plus important de la phase)
Seul endroit qui décide ce qu'un agent peut lire. Un document passe seulement si :
1. son type est dans la **liste blanche** (PAD, SAR, Program Document ; en repli : mémorandum du
   Président, fiche d'information « Appraisal Stage ») ;
2. son type n'est pas dans la **liste noire** (ICR, ICRR, PPAR, ISR...) et son titre ne contient
   pas de mot suspect (« Completion », « Restructuring »...) — double sécurité ;
3. il est en anglais ;
4. sa date est connue et **strictement antérieure** à l'approbation (ou, si la date d'approbation
   manque, au début de l'exercice d'approbation : borne prudente).

En cas de doute, le document est rejeté. *Tests écrits avant le code* (`tests/test_leakage.py`,
couverture 100 %), dont un faux PAD daté après l'approbation, un document daté le jour même, un
document sans date, chaque type interdit, et un test qui vérifie que listes blanche et noire ne se
chevauchent pas.

### `feasibility.py` — l'étude de faisabilité
- Applique le filtre aux 50 projets, télécharge réellement les textes gardés, classe chaque projet
  (A : dossier d'évaluation en texte ; B : document court seulement ; C : texte inutilisable ;
  D : rien d'ex ante).
- `wilson_ci()` : intervalle de confiance à 95 % de Wilson pour extrapoler 28/50 à 12 511 projets
  (plus fiable que la formule p ± 1,96·√(p(1−p)/n) quand n est petit).
- Recensement : liste tous les dossiers d'évaluation de l'API, applique le même filtre à tous les
  projets notés → compte réel, pas seulement une extrapolation.
- Écrit `reports/faisabilite_chiffres.md` (généré, ne pas éditer).

## Rapports (`reports/`)

| Fichier | Contenu |
| --- | --- |
| `faisabilite_chiffres.md` | Tous les chiffres bruts, générés par le code (détail des 50 projets inclus). |
| `faisabilite.md` | Le livrable : décision go / no-go, méthode, problèmes, incertitudes. |
| `explication_phase1.md` | Ce document. |

## Pour tout relancer de zéro

```bash
uv sync
uv run python -m investagents.data.ieg
uv run python -m investagents.data.projects_api
uv run python -m investagents.data.documents_api
uv run python -m investagents.data.feasibility
uv run pytest && uv run ruff check .
```

## Limite connue de la couverture de tests
`leakage.py` et `http.py` sont couverts à 100 % ; l'ensemble de `data/` à 64 %. Les parties non
couvertes sont les fonctions `main()` (téléchargement et affichage), testées en les exécutant
réellement, pas par des tests unitaires.
