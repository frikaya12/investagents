# Dépôts GitHub utiles — à étudier, pas à copier

**Statut : liste issue d'une recherche externe, NON VÉRIFIÉE.** Avant de s'appuyer sur un dépôt, vérifier
qu'il existe, lire son README, sa licence et sa date de dernière mise à jour. Ne reprendre du code que si la
licence le permet, et toujours avec attribution.

## Base principale (vérifiée)
| Dépôt | Ce qu'on en tire | Licence |
| --- | --- | --- |
| [pantageepapa/DIALECTIC](https://github.com/pantageepapa/DIALECTIC) | Boucle de débat : génération Pour/Contre, critique, notation, raffinement (`src/agent/pipeline/stages/`) ; graphe LangGraph ; schémas Pydantic. Papier : arXiv 2603.12274 (EACL 2026 Industry Track) | MIT, réutilisation autorisée par l'auteur |

## Proches de l'idée (à vérifier)
| Dépôt | Intérêt annoncé | Différence avec notre projet |
| --- | --- | --- |
| [ZhenchengLin/qfii](https://github.com/ZhenchengLin/qfii) | Agents spécialisés, comité agressif / conservateur / neutre, débats, vérification des preuves, LangGraph + MCP, validation humaine, tests du comité | Actions cotées et données de marché |
| [bostonbyte/investment-committee-swarm](https://github.com/bostonbyte/investment-committee-swarm) | Agents finance / technique / marché, Red Team, Managing Partner qui note | Startups |
| [lenamonj/equity-trader](https://github.com/lenamonj/equity-trader) | Huit agents qui convergent vers Buy / Hold / Sell | Actions en bourse |
| [sidnov6/quorum-investment-committee](https://github.com/sidnov6/quorum-investment-committee) | Agents Bull / Bear / Risk / Portfolio Manager, backtest avec protection contre le look-ahead bias | Portefeuille d'actifs financiers |

## Briques techniques (à vérifier)
| Dépôt | Ce qu'on peut regarder |
| --- | --- |
| [YairLutaty/multi-agent-debate](https://github.com/YairLutaty/multi-agent-debate) | Débat LangGraph, preuves, critique, évaluation |
| [slior/dialectic](https://github.com/slior/dialectic) | Propositions, critiques, raffinements, juge. **Attention : sans rapport avec le DIALECTIC de pantageepapa malgré le nom.** |
| [arepetti/madjury](https://github.com/arepetti/madjury) | Débat entre agents et arbitrage final |
| [longsizhuo/openInvest](https://github.com/longsizhuo/openInvest) | Traçabilité des décisions, backtesting |

## Ce qui nous distingue (aucun dépôt identifié ne le fait)
- Données IEG de la Banque mondiale comme vérité terrain, documents ex ante comme seule entrée.
- Comparaison chiffrée comité d'agents / gradient boosting / prompt unique, avec modèle hybride.
- Sélection sous budget et backtest de portefeuille sur des projets de développement.

## Ordre d'étude conseillé
1. DIALECTIC (notre base) : comprendre `graph.py` et `stages/`.
2. qfii : organisation LangGraph, preuves, validations humaines.
3. QUORUM : protection contre le look-ahead bias dans le backtest.
