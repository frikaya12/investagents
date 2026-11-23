"""Glossaire commun au rapport (md, html, pdf) et à la démo Streamlit.

Chaque entrée : terme, définition simple, définition technique, exemple « dans notre projet ».
Les exemples peuvent contenir des emplacements `{{clé}}` remplis par `metrics.load()` : aucun
chiffre n'est tapé à la main.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    term: str
    simple: str
    technique: str
    exemple: str

    @property
    def anchor(self) -> str:
        return slug(self.term)


def slug(text: str) -> str:
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return "g-" + re.sub(r"[^a-z0-9]+", "-", s).strip("-")


TERMS: list[Term] = [
    Term("AUC-ROC",
         "Note de 0 à 1 qui dit si le modèle range les projets en échec avant les projets réussis. "
         "0,5 = hasard, 1 = parfait.",
         "Aire sous la courbe ROC : probabilité qu'un projet en échec tiré au hasard reçoive un "
         "score plus élevé qu'un projet réussi tiré au hasard. Insensible au taux de base.",
         "Sur le coffre-fort, M1 obtient {{vault_m1_aucroc}} : l'intervalle contient 0,5, donc on "
         "ne distingue pas M1 du hasard."),
    Term("AUC-PR",
         "Note qui dit si, parmi les projets que le modèle signale comme risqués, beaucoup sont "
         "vraiment en échec. Le hasard vaut le taux de base, pas 0,5.",
         "Aire sous la courbe précision-rappel (average precision), calculée sur la classe "
         "minoritaire « échec ». Métrique principale du projet, car les échecs sont rares.",
         "M1-texte : {{m1t_aucpr}} au test, contre un taux de base de {{base_test_text}}."),
    Term("taux de base",
         "La part de projets en échec dans l'échantillon. C'est le score d'un modèle qui ne sait "
         "rien (M0).",
         "Prévalence de la classe positive ; valeur de l'AUC-PR d'un classement aléatoire.",
         "{{fail_train}} d'échecs à l'entraînement, {{fail_test}} au test, {{vault_fail}} sur le "
         "coffre-fort : c'est la dérive."),
    Term("Brier",
         "Erreur moyenne des probabilités annoncées. Plus c'est bas, mieux c'est.",
         "Moyenne de (p − y)² ; mesure à la fois le classement et la calibration.",
         "M1-texte : Brier {{brier_before}} avant calibration, {{brier_after}} après."),
    Term("ECE",
         "Écart moyen entre la probabilité annoncée et la fréquence réelle. Si le modèle dit "
         "« 20 % », il faut environ 20 % d'échecs.",
         "Expected Calibration Error : moyenne pondérée, par tranche de probabilité, de "
         "|fréquence observée − probabilité moyenne|.",
         "M1-texte : ECE {{ece_before}} avant la calibration de Platt, {{ece_after}} après."),
    Term("calibration de Platt",
         "Une petite correction qui rend les probabilités du modèle honnêtes, sans changer l'ordre "
         "des projets.",
         "Régression logistique à une variable (le score du modèle) apprise sur la validation "
         "seulement ; elle corrige aussi la dérive du taux d'échec.",
         "Recalibrée sur 2008–2011 avant la sélection sous budget, jamais sur le test."),
    Term("intervalle de confiance bootstrap",
         "La fourchette dans laquelle le vrai score a de bonnes chances de se trouver, vu le "
         "nombre limité de projets.",
         "IC à 95 % par percentiles sur 2 000 rééchantillonnages avec remise des projets du test.",
         "M1 au test : {{m1_aucpr}}."),
    Term("écart apparié",
         "La différence entre deux modèles mesurée sur exactement les mêmes projets, pour ne "
         "comparer que les modèles.",
         "Différence de métrique recalculée sur chaque tirage bootstrap commun aux deux modèles ; "
         "si l'IC contient 0, l'écart n'est pas démontré.",
         "M6 − M1-texte = {{m6_minus_text}} : zéro est dans l'intervalle."),
    Term("test de permutation",
         "On mélange au hasard les issues des projets pour voir quel score on obtiendrait par pure "
         "chance.",
         "Distribution nulle de la métrique obtenue en permutant les étiquettes (ou en tirant des "
         "portefeuilles au hasard sous le même budget) ; la valeur p est la part des tirages au "
         "moins aussi bons.",
         "Portefeuille à 25 % du budget : {{pf25_share}} de projets réussis, contre "
         "{{pf25_random}} en moyenne au hasard ({{pf25_p}})."),
    Term("valeur p",
         "La probabilité d'obtenir un résultat au moins aussi bon par pure chance. Petite = "
         "difficile à expliquer par le hasard.",
         "Part des tirages de la distribution nulle supérieurs ou égaux à la valeur observée.",
         "M1 contre des étiquettes permutées : {{m1_perm_p}}."),
    Term("validation croisée",
         "On découpe les données de réglage en morceaux, on apprend sur les uns et on vérifie sur "
         "l'autre, à tour de rôle.",
         "K-fold stratifié sur la seule période de validation, pour choisir un hyperparamètre sans "
         "toucher au test.",
         "M5/M6 réglés par validation croisée sur {{d_n_valid}} projets de validation."),
    Term("sur-apprentissage",
         "Le modèle apprend par cœur ses exemples et ne généralise pas.",
         "Écart entre la performance sur les données d'apprentissage et sur des données neuves ; "
         "on le limite par régularisation et réglage sur la validation.",
         "M1-texte : AUC-PR {{m1t_train_aucpr}} sur l'entraînement, {{m1t_valid_aucpr}} sur la "
         "validation."),
    Term("fuite de données",
         "Le modèle voit, sans le vouloir, une information qui n'existait pas au moment de la "
         "décision. Ses scores deviennent trop beaux.",
         "Toute variable ou tout texte postérieur à l'instant de prédiction (ici, l'approbation du "
         "projet) ; aussi l'usage du test pour faire des choix.",
         "Interdits : rapports de fin de projet (ICR, ICRR, PPAR), colonnes IEG post-projet, "
         "documents datés après l'approbation."),
    Term("découpage temporel",
         "On apprend sur le passé et on teste sur l'avenir, comme dans la vraie vie.",
         "Split par exercice d'approbation, sans mélange : entraînement ≤ 2007, validation "
         "2008–2011, test 2012–2017.",
         "{{n_train}} projets d'entraînement, {{n_valid}} de validation, {{n_test}} de test."),
    Term("coffre-fort",
         "Un paquet de projets mis sous clé dès le début et ouvert une seule fois, à la fin.",
         "Projets évalués par l'IEG en exercice ≥ 2025, exclus de tout entraînement, réglage et "
         "lecture ; protocole commité avant ouverture.",
         "{{vault_n}} projets ; M1 y tombe au niveau du hasard : {{vault_m1_aucpr}} pour "
         "un taux de "
         "base de {{vault_m0_aucpr}}."),
    Term("décalage de domaine",
         "Les données de demain ne ressemblent pas tout à fait à celles d'hier, et le modèle perd "
         "en qualité.",
         "Changement de la distribution des variables ou de la cible entre apprentissage et usage "
         "(ici, le taux d'échec baisse avec le temps).",
         "Taux d'échec {{fail_train}} (≤ 2007) contre {{vault_fail}} sur le coffre-fort."),
    Term("TF-IDF",
         "Une façon de transformer un texte en nombres en comptant les mots, en donnant plus de "
         "poids aux mots rares et parlants.",
         "Term frequency × inverse document frequency : mots seuls, mots vides anglais retirés, tf "
         "sous-linéaire, au plus 50 000 mots, vecteurs normalisés (L2).",
         "M1-texte = TF-IDF du dossier d'évaluation + régression logistique."),
    Term("régression logistique",
         "Un modèle simple qui additionne des indices pondérés pour donner une probabilité.",
         "Modèle linéaire sur le log-odds, régularisé (C), appris par maximum de vraisemblance.",
         "M1-texte, M3, M5, M6, M7 sont des régressions logistiques."),
    Term("gradient boosting",
         "Beaucoup de petits arbres de décision qui corrigent chacun les erreurs des précédents.",
         "HistGradientBoostingClassifier de scikit-learn : arbres ajoutés séquentiellement sur le "
         "gradient de la perte logistique.",
         "M1 : variables structurées ex ante (région, secteur, montant, indicateurs pays)."),
    Term("embeddings",
         "Une façon de résumer le sens d'un texte par une liste de nombres.",
         "Vecteurs denses produits par un modèle de langue ; deux textes proches ont des vecteurs "
         "proches.",
         "Variante MiniLM testée au bloc B sur la validation ; non retenue, moins bonne que le "
         "TF-IDF."),
    Term("LLM",
         "Un grand modèle de langue : un programme qui lit et écrit du texte.",
         "Large Language Model, réseau transformeur pré-entraîné ; ici Qwen2.5 7B, exécuté "
         "en local "
         "avec Ollama, sorties JSON validées par Pydantic.",
         "Aucun chiffre ne vient du LLM : les nombres sont comptés et vérifiés en Python."),
    Term("agent",
         "Un LLM à qui on donne un rôle précis (avocat Pour, avocat Contre, critique, juge) et des "
         "règles.",
         "Nœud d'un graphe de calcul qui appelle un LLM avec un rôle, un schéma de sortie et un "
         "état partagé.",
         "Le comité : Pour, Contre, critique, raffinement, Juge ; puis l'extracteur de facteurs."),
    Term("LangGraph",
         "La bibliothèque qui enchaîne les agents dans le bon ordre.",
         "Bibliothèque de graphes d'état pour orchestrer des appels LLM (nœuds, arêtes, état "
         "typé).",
         "Débat → critique → raffinement → Juge → vérification des citations."),
    Term("citation vérifiée",
         "Chaque argument doit citer une phrase du dossier, et on vérifie par programme que cette "
         "phrase existe.",
         "Citations par numéros de phrases [S12] ; contrôle en Python que le numéro existe et que "
         "la phrase se retrouve mot pour mot dans le texte source.",
         "{{com_quotes}} citations retrouvées ({{com_quotes_pct}})."),
    Term("test de reconnaissance",
         "On demande au LLM s'il reconnaît le projet. S'il le reconnaît, il pourrait tricher avec "
         "sa mémoire.",
         "Questions directes (nom, pays, année, note IEG) sur des dossiers masqués ; réponses "
         "comparées à la vérité en local.",
         "{{rec_recognized}} projet reconnu, {{rec_rating}} note IEG proposée."),
    Term("sac à dos (CP-SAT)",
         "Choisir le meilleur panier de projets sans dépasser le budget.",
         "Problème du sac à dos 0-1 résolu exactement par le solveur CP-SAT d'OR-Tools, avec un "
         "plafond par pays.",
         "Objectif figé : maximiser Σ p × coût, 25 % du budget, au plus 25 % par pays."),
    Term("IEG",
         "Le service d'évaluation indépendant de la Banque mondiale. Il note chaque projet après "
         "sa clôture.",
         "Independent Evaluation Group ; note de résultat sur 6 niveaux (HS, S, MS, MU, U, HU).",
         "Vérité terrain : échec = MU, U ou HU."),
    Term("PAD",
         "Le dossier d'évaluation écrit avant l'approbation du projet.",
         "Project Appraisal Document (ou Staff Appraisal Report pour les anciens projets) : "
         "document ex ante lu par les agents.",
         "{{n_exploitable}} projets ont un dossier en texte daté avant l'approbation."),
    Term("critères OCDE-CAD",
         "Une grille internationale pour juger un projet de développement.",
         "Critères d'évaluation du Comité d'aide au développement de l'OCDE : pertinence, "
         "efficience, efficacité, impact, durabilité (et cohérence).",
         "Le Juge note 5 critères inspirés de cette grille, de 0 à 10."),
    Term("frontière α",
         "Un curseur qui dit combien on fait confiance au ML et combien au jugement du comité.",
         "Score = α × probabilité de réussite + (1 − α) × moyenne pondérée des 5 notes / 10 ; "
         "α ∈ {1 ; 0,75 ; 0,5 ; 0,25 ; 0} fixés à l'avance.",
         "Au test, de α = 1 à α = 0, la note d'impact du portefeuille passe de "
         "{{frt_1_0_impact}} à "
         "{{frt_0_0_impact}}."),
]

BY_KEY = {t.term.lower(): t for t in TERMS}


def find(term: str) -> Term:
    try:
        return BY_KEY[term.lower()]
    except KeyError as e:
        raise KeyError(f"terme absent du glossaire : {term}") from e
