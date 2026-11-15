# Prompts à envoyer à Claude Code, un par un (attendre la fin de chacun)

## Prompt 1 — Comprendre
Salut ! On démarre InvestAgents. Lis `CLAUDE.md`, puis `docs/cahier_des_charges_v3.md` (ou le PDF dans `docs/`)
et `docs/references_github.md`. Résume-moi en 5 lignes le projet, les règles anti-fuite et l'ordre des phases,
pour que je vérifie que tu as compris. Ne code rien.

## Prompt 2 — Plan de la phase 1
Propose-moi un plan pour la phase 1 (faisabilité) uniquement : fichiers à créer, ce que fait chacun, et à quoi
ressemblera `reports/faisabilite.md`. Pas de LangGraph, pas d'agents. Attends mon accord.

## Prompt 3 — Squelette du dépôt
Crée le squelette : `uv init`, Ruff, pytest, `.gitignore` (data/raw non versionné), la structure de dossiers du
cahier des charges (dossiers vides avec `__init__.py`), un README court. Explique-moi chaque fichier créé.

## Prompt 4 — Données IEG
Écris `src/investagents/data/ieg.py` : télécharge le jeu « IEG World Bank Project Performance Ratings »,
sauvegarde-le en Parquet, et affiche : nombre de projets, répartition des notes de résultat, répartition par
année d'évaluation. Vérifie le vrai format des colonnes avant de coder le nettoyage. Ajoute des tests.

## Prompt 5 — Documents ex ante sur 50 projets
Écris `src/investagents/data/documents_api.py` : pour 50 projets notés tirés au hasard (graine fixe), interroge
l'API Documents & Reports v3 et liste les documents (type, date, `txturl`). Trouve par des appels réels quelle
valeur de type de document correspond au dossier d'évaluation ex ante, et comment filtrer par projet.
Ne devine pas les paramètres : teste-les et montre-moi les réponses.

## Prompt 6 — Filtre anti-fuite
Écris `src/investagents/data/leakage.py` : ne garde que les documents ex ante datés avant l'approbation du
projet (date d'approbation via la Projects API). Écris d'abord les tests, y compris avec un faux document
post-approbation qui doit être rejeté.

## Prompt 7 — Rapport de faisabilité
Produis `reports/faisabilite.md` : sur les 50 projets, combien ont un document ex ante en texte ? Extrapole
prudemment au jeu complet avec un intervalle de confiance. Donne une recommandation go / no-go (seuil : 300
projets) et les problèmes rencontrés.
