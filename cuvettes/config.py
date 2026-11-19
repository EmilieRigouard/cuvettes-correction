"""Configuration : chemins lus dans le .env, parametres modifiables ici."""

import os
from pathlib import Path

from dotenv import load_dotenv

RACINE = Path(__file__).resolve().parent.parent
load_dotenv(RACINE / ".env")


def _chemin(nom, obligatoire=True):
    valeur = os.getenv(nom, "").strip().strip('"').strip("'")
    if not valeur:
        if obligatoire:
            raise SystemExit(
                "Variable " + nom + " manquante dans le fichier .env "
                "(copie .env.example en .env et remplis-le)."
            )
        return None
    return Path(valeur)


# --- Chemins (depuis le .env) ---
MODELE_YOLO = _chemin("MODELE_YOLO")
DOSSIER_IMAGES = _chemin("DOSSIER_IMAGES")
DOSSIER_RESULTATS = _chemin("DOSSIER_RESULTATS")
MODELE_CLASSIFIEUR = _chemin("MODELE_CLASSIFIEUR", obligatoire=False)
DEVICE = os.getenv("DEVICE", "cpu").strip()

CSV_CORRIGE = DOSSIER_RESULTATS / "detections_corrigees.csv"
DOSSIER_IMAGES_CORRIGEES = DOSSIER_RESULTATS / "images"
FICHIER_VALIDES = DOSSIER_RESULTATS / "images_validees.txt"

# --- Detection ---
SEUIL_INFERENCE = 0.25
TAILLE_TUILE = 1024
OVERLAP = 0.2
PASSE_IMAGE_ENTIERE = True
FUSION_TYPE = "NMS"
FUSION_SEUIL = 0.4

# --- Filtres par regles ---
FILTRES = {
    "score_min": 0.40,
    "taille_min": 30,
    "taille_max": 400,
    "ratio_max": 1.5,
    "contraste_min": None,
    "contraste_max": None,
}
SEUIL_RF = 0.5

# --- Interface ---
FENETRE_MAX_W = 1500
FENETRE_MAX_H = 850
RAYON_DEFAUT = 20  
ZOOM_MAX = 12.0     
ZOOM_PAS = 1.25  
   
# --- Colonnes du CSV ---
FEATURES = ["score", "largeur", "hauteur", "ratio", "contraste",
            "moy_int", "std_int"]
COLONNES = (["id", "image", "x1", "y1", "x2", "y2"] + FEATURES +
            ["garde_regles", "raison_rejet", "proba_rf", "garde_final",
             "garde_manuel", "ajout_manuel", "label"])


def verifier():
    """Verifie les chemins avant de demarrer."""
    if not MODELE_YOLO.is_file():
        raise SystemExit("Modele YOLO introuvable : " + str(MODELE_YOLO))
    if not DOSSIER_IMAGES.is_dir():
        raise SystemExit("Dossier d'images introuvable : "
                         + str(DOSSIER_IMAGES))
    if MODELE_CLASSIFIEUR is not None and not MODELE_CLASSIFIEUR.is_file():
        raise SystemExit("Classifieur introuvable : "
                         + str(MODELE_CLASSIFIEUR))
    DOSSIER_RESULTATS.mkdir(parents=True, exist_ok=True)