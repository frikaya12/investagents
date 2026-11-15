# Faisabilité : chiffres bruts (fichier généré automatiquement)

Généré par `uv run python -m investagents.data.feasibility`. Ne pas éditer à la main.

- Projets IEG notés (cible binaire connue) : **12511**
- Seuil de texte exploitable : 10 000 caractères

## 1. Échantillon de 50 projets notés (graine 42)

| Statut | Projets |
| --- | ---: |
| Dossier d'évaluation (tier 1) avec texte exploitable | 28 |
| Seulement un document ex ante court (tier 2) avec texte exploitable | 6 |
| Document ex ante présent mais texte absent ou inutilisable | 1 |
| Aucun document ex ante daté avant l'approbation | 15 |
| **Total** | **50** |

- Part avec dossier d'évaluation en texte (A) : 28/50 = 56%, IC 95 % Wilson [42% ; 69%]
- Extrapolation à 12511 projets notés : 7 006 projets [5 293 ; 8 612]
- Avec le repli tier 2 (A + B) : 34/50 = 68%  IC 95 % [54% ; 79%] -> 8 507 projets [6 780 ; 9 914]
- Documents tier 1 gardés avec un txturl : 30 ; dont texte ≥ seuil après téléchargement : 29

Détail par projet :

| Projet | Exercice | Note IEG | Statut | Docs ex ante | Longueur max (car.) |
| --- | ---: | --- | --- | ---: | ---: |
| P003155 | 1969 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P004042 | 1969 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P009667 | 1972 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P010116 | 1978 | Satisfactory | B_tier2_text | 2 | 226 602 |
| P000803 | 1978 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P004704 | 1979 | Unsatisfactory | A_tier1_text | 2 | 198 258 |
| P007355 | 1979 | Satisfactory | A_tier1_text | 1 | 236 941 |
| P007969 | 1981 | Unsatisfactory | B_tier2_text | 1 | 103 674 |
| P004896 | 1981 | Unsatisfactory | A_tier1_text | 2 | 134 503 |
| P008952 | 1984 | Satisfactory | A_tier1_text | 1 | 186 377 |
| P001443 | 1984 | Satisfactory | A_tier1_text | 2 | 132 690 |
| P002593 | 1985 | Satisfactory | A_tier1_text | 2 | 241 782 |
| P010220 | 1985 | Satisfactory | A_tier1_text | 2 | 261 132 |
| P003434 | 1985 | Moderately Satisfactory | A_tier1_text | 2 | 245 435 |
| P004744 | 1986 | Satisfactory | B_tier2_text | 1 | 215 900 |
| P001764 | 1987 | Satisfactory | A_tier1_text | 1 | 256 068 |
| P010307 | 1988 | Unsatisfactory | A_tier1_text | 2 | 260 504 |
| P001653 | 1988 | Satisfactory | B_tier2_text | 1 | 141 463 |
| P010304 | 1988 | Satisfactory | A_tier1_text | 2 | 216 964 |
| P002750 | 1989 | Satisfactory | B_tier2_text | 1 | 208 545 |
| P008133 | 1990 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P007647 | 1991 | Moderately Unsatisfactory | B_tier2_text | 2 | 37 862 |
| P000931 | 1992 | Moderately Satisfactory | D_no_ex_ante | 0 | 0 |
| P010476 | 1995 | Satisfactory | A_tier1_text | 3 | 798 746 |
| P006512 | 1996 | Unsatisfactory | A_tier1_text | 2 | 375 270 |
| P047065 | 1997 | Unsatisfactory | C_ex_ante_no_text | 1 | 0 |
| P039029 | 1997 | Satisfactory | A_tier1_text | 2 | 186 303 |
| P055304 | 1999 | Satisfactory | A_tier1_text | 1 | 185 690 |
| P050668 | 2002 | Moderately Satisfactory | A_tier1_text | 1 | 378 995 |
| P049621 | 2002 | Moderately Satisfactory | A_tier1_text | 1 | 331 274 |
| P070191 | 2003 | Moderately Satisfactory | A_tier1_text | 1 | 516 006 |
| P104716 | 2008 | Moderately Satisfactory | D_no_ex_ante | 0 | 0 |
| P110167 | 2008 | Moderately Satisfactory | A_tier1_text | 1 | 240 042 |
| P111840 | 2010 | Moderately Satisfactory | A_tier1_text | 1 | 444 289 |
| P082144 | 2010 | Highly Satisfactory | A_tier1_text | 1 | 423 901 |
| P121231 | 2011 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P113976 | 2011 | Moderately Unsatisfactory | D_no_ex_ante | 0 | 0 |
| P121289 | 2012 | Satisfactory | A_tier1_text | 2 | 117 390 |
| P115954 | 2012 | Satisfactory | A_tier1_text | 2 | 251 513 |
| P094183 | 2013 | Moderately Satisfactory | A_tier1_text | 2 | 330 408 |
| P120397 | 2013 | Moderately Satisfactory | D_no_ex_ante | 0 | 0 |
| P143274 | 2014 | Moderately Unsatisfactory | A_tier1_text | 2 | 235 646 |
| P132786 | 2014 | Moderately Unsatisfactory | D_no_ex_ante | 0 | 0 |
| P147006 | 2014 | Satisfactory | A_tier1_text | 2 | 342 712 |
| P130847 | 2014 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P148183 | 2015 | Moderately Satisfactory | D_no_ex_ante | 0 | 0 |
| P133777 | 2015 | Satisfactory | D_no_ex_ante | 0 | 0 |
| P148531 | 2017 | Satisfactory | A_tier1_text | 2 | 213 536 |
| P173906 | 2020 | Moderately Satisfactory | D_no_ex_ante | 0 | 0 |
| P176663 | 2022 | Satisfactory | A_tier1_text | 1 | 272 834 |

## 2. Recensement complet des dossiers d'évaluation (tier 1)

- Documents tier 1 listés par l'API : 15142 lignes document-projet
- Projets notés avec ≥ 1 dossier tier 1 en anglais, daté avant l'approbation, avec txturl : **8260** (66% des notés)
- Référence de date utilisée : {'board_approval_date': 8164, 'fiscal_year_start': 96}
- Dossiers tier 1 de projets notés datés le jour même de l'approbation (rejetés) : 119 ; datés après (rejetés) : 889
- Sensibilité : projets en plus si l'on tolérait un dossier daté ≤ 31 jours après l'approbation (NON appliqué) : 921
- Projets notés sans date d'approbation exacte dans la Projects API : 1803
- Estimation après contrôle du texte (taux de l'échantillon 29/30) : 7 985 projets

| Exercice d'approbation   |   notés |   avec dossier tier 1 (txturl) | part   |
|:-------------------------|--------:|-------------------------------:|:-------|
| ≤ 1997                   |    6259 |                           3852 | 62%    |
| 1998–2007                |    2745 |                           2020 | 74%    |
| 2008–2015                |    2351 |                           1541 | 66%    |
| ≥ 2016                   |    1156 |                            847 | 73%    |

Répartition des notes parmi ces projets :

| Note IEG                  |   projets |
|:--------------------------|----------:|
| Satisfactory              |      3636 |
| Moderately Satisfactory   |      2183 |
| Unsatisfactory            |      1218 |
| Moderately Unsatisfactory |       857 |
| Highly Satisfactory       |       231 |
| Highly Unsatisfactory     |       135 |

- Non satisfaisants (MU/U/HU) : 2210 ; satisfaisants : 6050
- Dont évalués par l'IEG en exercice ≥ 2025 : 550
