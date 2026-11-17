"""Lecture / ecriture du CSV, des images validees et des images annotees."""

import csv

import cv2

from . import config


def lister_images():
    noms = sorted(p.name for p in config.DOSSIER_IMAGES.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg"))
    if not noms:
        raise SystemExit("Aucune image .jpg dans "
                         + str(config.DOSSIER_IMAGES))
    return noms


def centre_rayon(l):
    cx = (l["x1"] + l["x2"]) // 2
    cy = (l["y1"] + l["y2"]) // 2
    r = max(l["x2"] - l["x1"], l["y2"] - l["y1"]) // 2
    return cx, cy, max(r, 3)


class Donnees:
    def __init__(self):
        self.data = {}        # nom image -> liste de detections
        self.valides = set()  # images validees
        ids = []

        if config.CSV_CORRIGE.exists():
            with open(config.CSV_CORRIGE, newline="",
                      encoding="utf-8-sig") as f:
                for l in csv.DictReader(f, delimiter=";"):
                    for k in ("x1", "y1", "x2", "y2"):
                        l[k] = int(float(l[k]))
                    l["garde_manuel"] = int(l.get("garde_manuel") or 0)
                    l["ajout_manuel"] = int(l.get("ajout_manuel") or 0)
                    self.data.setdefault(l["image"], []).append(l)
                    ids.append(int(float(l["id"])))

        if config.FICHIER_VALIDES.exists():
            texte = config.FICHIER_VALIDES.read_text(encoding="utf-8")
            self.valides = set(x.strip() for x in texte.splitlines()
                               if x.strip())

        # une image validee sans cuvette ne doit pas etre re-detectee
        for nom in self.valides:
            self.data.setdefault(nom, [])

        self.next_id = max(ids, default=0) + 1

    def nouvel_id(self):
        n = self.next_id
        self.next_id += 1
        return n

    def sauver(self):
        with open(config.CSV_CORRIGE, "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=config.COLONNES,
                               delimiter=";", extrasaction="ignore")
            w.writeheader()
            for nom in self.data:
                for l in self.data[nom]:
                    l["label"] = l["garde_manuel"]
                    w.writerow(l)
        config.FICHIER_VALIDES.write_text(
            "\n".join(sorted(self.valides)), encoding="utf-8")

    def sauver_image(self, nom):
        img = cv2.imread(str(config.DOSSIER_IMAGES / nom))
        if img is None:
            return
        for l in self.data[nom]:
            if l["garde_manuel"]:
                cx, cy, r = centre_rayon(l)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), 6)
        config.DOSSIER_IMAGES_CORRIGEES.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.DOSSIER_IMAGES_CORRIGEES / nom), img)

    def valider(self, nom):
        self.valides.add(nom)
        self.sauver_image(nom)
        self.sauver()