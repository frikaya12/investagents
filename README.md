# InvestAgents

Comité d'agents IA pour l'évaluation multicritère et la sélection de projets d'investissement.
Les agents lisent les documents d'évaluation **ex ante** de projets financés par la Banque mondiale ;
la vérité terrain est la note de résultat de l'IEG. Spécification : `docs/Cahier_des_charges_v3_InvestAgents.pdf`.

**Phase en cours : 1, faisabilité** (scripts de données et comptes, pas encore d'agents).

## Installation

```bash
uv sync                 # crée .venv avec Python 3.12 et installe les dépendances
uv run pytest           # tests hors réseau
uv run pytest -m network  # tests qui appellent les vraies API
uv run ruff check .     # lint
```

## Structure

```
data/raw/          données brutes téléchargées (non versionnées)
data/processed/    Parquet régénérés par les scripts (non versionnés)
src/investagents/  code : data/, schemas/, agents/, debate/, graph/, ml/, selection/, app/
tests/             tests pytest
reports/           rapports (faisabilité, explications)
experiments/       configurations et résultats des expériences
```

## Données et licences

- IEG World Bank Project Performance Ratings : © Banque mondiale, licence CC BY 4.0.
- World Bank Projects API et Documents & Reports API : données publiques de la Banque mondiale.
- Code adapté de DIALECTIC (github.com/pantageepapa/DIALECTIC), licence MIT (à partir de la phase 3).
