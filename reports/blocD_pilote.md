# Bloc D, pilote : lecture (écrit à la main)

- **Les 3 critères fixés à l'avance sont remplis** : citations valides 99,8 % ; 27 s par projet
  (médiane, un GPU T4) ; 19 facteurs sur 19 trouvés dans plus de 30 % des dossiers ; 0 erreur sur
  30 projets. Le passage complet est donc lancé (validation, puis test).
- **Limite vue au contrôle manuel (4 projets × 3 facteurs).** Une citation « valide » veut dire
  que la phrase existe et parle du sujet (99,5 % contiennent un mot-clé du facteur). Cela ne veut
  pas dire que la réponse est juste. Sur first_operation, 2 « yes » sur 3 sont faux : le modèle
  lit « the first project » (un projet précédent) comme « première opération ». Pilot_innovative et
  weak_capacity sont plausibles. La grille étant figée, elle n'est pas modifiée ; M6 dira si ces
  variables portent de l'information malgré ce bruit.
- Plusieurs facteurs oui / non sont presque toujours « yes » quand ils sont trouvés
  (pilot_innovative 29/29, weak_capacity 19/19) : l'information tient alors surtout à « trouvé /
  non trouvé ».

# Bloc D : extracteur — contrôle (experiments/blocD/kaggle_pilot) (généré)

- Projets : 30 réussis / 30 ; erreurs 0.
- Citations valides : 99.8% (critère ≥ 95 %) ; phrase citée contenant un mot-clé du facteur : 99.5%.
- Temps par projet (un GPU) : médiane 27 s, 90e centile 31 s (critère ≤ 40 s).
- Facteurs trouvés dans plus de 30 % des dossiers : 19 / 19 (critère ≥ 10).
- **Pilote bon : oui** {'citations>=95%': True, 'median<=40s': True, '>=10 factors found in >30%': True}

| Facteur | type | trouvé | incohérent | citations valides | mot-clé | valeurs |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| components | list | 100% | 0 | 97% | 97% | {3: 9, 2: 9, 1: 7, 4: 4} |
| implementing_agencies | list | 97% | 0 | 100% | 100% | {1: 15, 2: 4, 3: 4, 6: 2} |
| cofinanciers | list | 80% | 1 | 100% | 96% | {2: 11, 1: 8, 5: 2, 3: 2} |
| first_operation | yes_no | 47% | 0 | 100% | 100% | {'yes': 10, 'no': 4} |
| follow_on_phase | yes_no | 57% | 0 | 100% | 100% | {'yes': 16, 'no': 1} |
| reform_to_adopt | yes_no | 80% | 0 | 100% | 100% | {'yes': 23, 'no': 1} |
| env_category | category | 73% | 0 | 100% | 100% | {'B': 12, 'C': 6, 'A': 3, 'FI': 1} |
| overall_risk_rating | category | 73% | 0 | 100% | 100% | {'moderate': 12, 'high': 6, 'substantial': 3, 'low': 1} |
| fiduciary_risk | category | 73% | 0 | 100% | 100% | {'moderate': 8, 'substantial': 7, 'high': 6, 'low': 1} |
| baseline_values | yes_no | 83% | 0 | 100% | 100% | {'yes': 24, 'no': 1} |
| lessons_problems | yes_no | 60% | 0 | 100% | 100% | {'yes': 15, 'no': 3} |
| fragility_conflict | yes_no | 53% | 0 | 100% | 100% | {'yes': 15, 'no': 1} |
| weak_capacity | yes_no | 63% | 0 | 100% | 100% | {'yes': 19} |
| counterpart_funding | yes_no | 50% | 0 | 100% | 100% | {'yes': 15} |
| resettlement | yes_no | 77% | 0 | 100% | 100% | {'yes': 12, 'no': 11} |
| decentralized_implementation | yes_no | 90% | 0 | 100% | 100% | {'yes': 26, 'no': 1} |
| pilot_innovative | yes_no | 97% | 0 | 100% | 100% | {'yes': 29} |
| political_risk | yes_no | 77% | 0 | 100% | 100% | {'yes': 23} |
| economic_return_reported | yes_no | 70% | 0 | 100% | 100% | {'yes': 18, 'no': 3} |
