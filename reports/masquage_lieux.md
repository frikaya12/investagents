# Masquage des lieux : noms restants avant / après la liste fixe (généré)

Échantillon : 300 projets tirés au hasard dans l'entraînement (graine 5),
sections clés (≈ 8 000 caractères). v1 = spaCy + pays des relevés IDA/IBRD ;
v2 = v1 + liste fixe GeoNames (pays, capitales, accents ignorés) + anciens noms.

| Compteur | brut | v1 | v2 | réduction v1 → v2 |
| --- | ---: | ---: | ---: | ---: |
| pays + capitales | 1352 | 110 | 0 | 100% |
| autres villes ≥ 100 000 hab. (indépendant) | 561 | 328 | 324 | 1% |

Jetons [PLACE] insérés : v1 = 2904, v2 = 3017.

Noms encore présents en v2 (10 plus fréquents) :

- pays + capitales : []
- autres villes ≥ 100 000 hab. (indépendant) : [('Izmir', 15), ('Parana', 14), ('La Paz', 12), ('Bobo-Dioulasso', 10), ('Valencia', 9), ('Cochabamba', 8), ('Cotonou', 8), ('Sector 1', 7), ('Busan', 7), ('Gdansk', 7)]

Lecture : le compteur « pays + capitales » est presque nul en v2 par construction ;
le compteur indépendant (autres villes) dit ce que le masquage laisse passer.

**Décision (validée par Aya) :** on garde le masquage v2. Limite assumée : environ 1 grande
ville (hors capitales) reste visible par document (324 mentions pour 300 projets), contre une
dizaine de [PLACE] par document. Le masquage réduit l'information pays sans la supprimer.
