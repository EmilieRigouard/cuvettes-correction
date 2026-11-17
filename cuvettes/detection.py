"""Detection YOLO + SAHI, calcul des features et filtres."""

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from . import config


def calculer_features(x1, y1, x2, y2, score, gray):
    H, W = gray.shape
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))
    w = x2 - x1
    h = y2 - y1
    feats = {"score": float(score), "largeur": w, "hauteur": h,
             "ratio": max(w, h) / max(1, min(w, h)),
             "contraste": 0.0, "moy_int": 0.0, "std_int": 0.0}
    if w < 2 or h < 2:
        return feats, (x1, y1, x2, y2)

    dedans = gray[y1:y2, x1:x2].astype(float)
    feats["moy_int"] = float(dedans.mean())
    feats["std_int"] = float(dedans.std())

    mx = int(w * 0.5)
    my = int(h * 0.5)
    X1 = max(0, x1 - mx)
    Y1 = max(0, y1 - my)
    X2 = min(W, x2 + mx)
    Y2 = min(H, y2 + my)
    zone = gray[Y1:Y2, X1:X2].astype(float)
    masque = np.ones(zone.shape, dtype=bool)
    masque[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1] = False
    if masque.any():
        feats["contraste"] = float(dedans.mean() - zone[masque].mean())
    return feats, (x1, y1, x2, y2)


def passe_les_regles(f):
    F = config.FILTRES
    taille = max(f["largeur"], f["hauteur"])
    if f["score"] < F["score_min"]:
        return False, "score"
    if not (F["taille_min"] <= taille <= F["taille_max"]):
        return False, "taille"
    if f["ratio"] > F["ratio_max"]:
        return False, "ratio"
    if F["contraste_min"] is not None:
        if f["contraste"] < F["contraste_min"]:
            return False, "contraste"
    if F["contraste_max"] is not None:
        if f["contraste"] > F["contraste_max"]:
            return False, "contraste"
    return True, ""


class Detecteur:
    def __init__(self):
        print("Chargement du modele YOLO...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(config.MODELE_YOLO),
            confidence_threshold=config.SEUIL_INFERENCE,
            device=config.DEVICE,
        )
        self.rf = None
        if config.MODELE_CLASSIFIEUR is not None:
            import joblib
            self.rf = joblib.load(config.MODELE_CLASSIFIEUR)
            print("Classifieur charge.")

    def detecter(self, nom):
        """Retourne la liste des detections (dicts) d'une image."""
        chemin = str(config.DOSSIER_IMAGES / nom)
        result = get_sliced_prediction(
            chemin,
            self.model,
            slice_height=config.TAILLE_TUILE,
            slice_width=config.TAILLE_TUILE,
            overlap_height_ratio=config.OVERLAP,
            overlap_width_ratio=config.OVERLAP,
            postprocess_type=config.FUSION_TYPE,
            postprocess_match_metric="IOS",
            postprocess_match_threshold=config.FUSION_SEUIL,
            perform_standard_pred=config.PASSE_IMAGE_ENTIERE,
        )
        img_couleur = cv2.imread(chemin)
        gray = cv2.cvtColor(img_couleur, cv2.COLOR_BGR2GRAY)

        rows = []
        for pred in result.object_prediction_list:
            b = pred.bbox
            feats, box = calculer_features(
                b.minx, b.miny, b.maxx, b.maxy, pred.score.value, gray)
            ok, raison = passe_les_regles(feats)
            proba = ""
            garde = ok
            if ok and self.rf is not None:
                X = [[feats[k] for k in config.FEATURES]]
                proba = round(float(self.rf.predict_proba(X)[0][1]), 3)
                garde = proba >= config.SEUIL_RF
                if not garde:
                    raison = "rf"
            row = {c: "" for c in config.COLONNES}
            row["image"] = nom
            row["x1"], row["y1"], row["x2"], row["y2"] = box
            for k in config.FEATURES:
                row[k] = round(feats[k], 3)
            row["garde_regles"] = int(ok)
            row["raison_rejet"] = raison
            row["proba_rf"] = proba
            row["garde_final"] = int(garde)
            row["garde_manuel"] = int(garde)
            row["ajout_manuel"] = 0
            rows.append(row)
        return rows