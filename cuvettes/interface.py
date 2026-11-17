"""Fenetre de correction : detection + suppression / ajout de cuvettes."""

import statistics
from concurrent.futures import ThreadPoolExecutor

import cv2

from . import config
from .detection import Detecteur
from .donnees import Donnees, centre_rayon, lister_images

FENETRE = "Correction"


class Application:
    def __init__(self):
        config.verifier()
        self.noms = lister_images()
        self.donnees = Donnees()
        self.detecteur = Detecteur()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}

        self.i = 0
        self.nom = None
        self.img = None
        self.echelle = 1.0
        self.dirty = True
        self.historique = []
        self.drag = None
        self.rayon = config.RAYON_SECOURS
        self.rayon_manuel = False

        print(len(self.noms), "images,",
              len(self.donnees.valides), "deja validees.")

    # ----- detection en arriere-plan -----
    def demander(self, nom):
        if nom not in self.donnees.data and nom not in self.futures:
            self.futures[nom] = self.pool.submit(
                self.detecteur.detecter, nom)

    def obtenir(self, nom):
        data = self.donnees.data
        if nom not in data:
            self.demander(nom)
            if not self.futures[nom].done():
                print("Detection en cours :", nom, "...")
            rows = self.futures.pop(nom).result()
            for r in rows:
                r["id"] = self.donnees.nouvel_id()
            data[nom] = rows
        return data[nom]

    # ----- navigation -----
    def charger(self, i):
        nom = self.noms[i]
        dets = self.obtenir(nom)
        if i + 1 < len(self.noms):
            self.demander(self.noms[i + 1])
        img = cv2.imread(str(config.DOSSIER_IMAGES / nom))
        if img is None:
            raise SystemExit("Image illisible : " + nom)
        h, w = img.shape[:2]
        self.i = i
        self.nom = nom
        self.img = img
        self.echelle = min(config.FENETRE_MAX_W / w,
                           config.FENETRE_MAX_H / h, 1.0)
        self.historique = []
        self.drag = None
        self.dirty = True
        if not self.rayon_manuel:
            r = [centre_rayon(l)[2] for l in dets if l["garde_manuel"]]
            if r:
                self.rayon = int(statistics.median(r))

    def premiere_non_validee(self):
        for k, nom in enumerate(self.noms):
            if nom not in self.donnees.valides:
                return k
        print("Toutes les images sont deja validees.")
        return 0

    # ----- edition -----
    def ajouter_cuvette(self, cx, cy, r):
        h, w = self.img.shape[:2]
        r = max(int(r), 3)
        l = {c: "" for c in config.COLONNES}
        l["id"] = self.donnees.nouvel_id()
        l["image"] = self.nom
        l["x1"] = max(0, int(cx - r))
        l["y1"] = max(0, int(cy - r))
        l["x2"] = min(w, int(cx + r))
        l["y2"] = min(h, int(cy + r))
        l["garde_regles"] = 1
        l["garde_final"] = 1
        l["garde_manuel"] = 1
        l["ajout_manuel"] = 1
        self.donnees.data[self.nom].append(l)
        self.historique.append(("ajout", l))
        self.dirty = True

    def annuler(self):
        if not self.historique:
            return
        action, l = self.historique.pop()
        if action == "toggle":
            l["garde_manuel"] = 1 - l["garde_manuel"]
        else:
            self.donnees.data[self.nom].remove(l)
        self.dirty = True

    # ----- affichage -----
    def dessiner(self):
        s = self.echelle
        vue = cv2.resize(self.img, None, fx=s, fy=s,
                         interpolation=cv2.INTER_AREA)
        dets = self.donnees.data[self.nom]
        for l in dets:
            cx, cy, r = centre_rayon(l)
            cx = int(cx * s)
            cy = int(cy * s)
            r = max(int(r * s), 3)
            if l["garde_manuel"] and l["ajout_manuel"]:
                cv2.circle(vue, (cx, cy), r, (0, 200, 0), 3)
            elif l["garde_manuel"]:
                cv2.circle(vue, (cx, cy), r, (0, 0, 255), 3)
            else:
                cv2.circle(vue, (cx, cy), r, (0, 165, 255), 1)
        if self.drag:
            x0, y0, x1, y1 = self.drag
            rr = int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
            cv2.circle(vue, (x0, y0), max(rr, 3), (255, 255, 0), 2)

        nb = sum(l["garde_manuel"] for l in dets)
        t1 = "[%d/%d] %s - %d cuvettes" % (
            self.i + 1, len(self.noms), self.nom, nb)
        t2 = "clic G=supprimer  clic D=ajouter (taille %d, +/-)" % self.rayon
        t3 = "d=valider+suivant a=precedent z=annuler q=quitter"
        cv2.rectangle(vue, (0, 0), (vue.shape[1], 50), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vue, t1 + "   " + t3, (8, 18), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vue, t2, (8, 40), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return vue

    def souris(self, event, x, y, flags, param):
        s = self.echelle

        if event == cv2.EVENT_LBUTTONDOWN:
            X = x / s
            Y = y / s
            meilleur = None
            meilleur_score = None
            for l in self.donnees.data[self.nom]:
                cx, cy, r = centre_rayon(l)
                d = ((X - cx) ** 2 + (Y - cy) ** 2) ** 0.5
                tol = max(r, 20 / s)
                if d <= tol:
                    score = d / tol
                    if meilleur is None or score < meilleur_score:
                        meilleur = l
                        meilleur_score = score
            if meilleur is not None:
                meilleur["garde_manuel"] = 1 - meilleur["garde_manuel"]
                self.historique.append(("toggle", meilleur))
                self.dirty = True

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drag = (x, y, x, y)
            self.dirty = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            x0, y0, _, _ = self.drag
            self.drag = (x0, y0, x, y)
            self.dirty = True

        elif event == cv2.EVENT_RBUTTONUP and self.drag:
            x0, y0, _, _ = self.drag
            dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            self.drag = None
            r = self.rayon if dist < 8 else dist / s
            self.ajouter_cuvette(x0 / s, y0 / s, r)

    # ----- boucle principale -----
    def lancer(self):
        self.charger(self.premiere_non_validee())
        cv2.namedWindow(FENETRE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(FENETRE, self.souris)

        while True:
            if self.dirty:
                cv2.imshow(FENETRE, self.dessiner())
                self.dirty = False

            k = cv2.waitKey(30) & 0xFF

            if k in (ord("d"), 32, 13):
                self.donnees.valider(self.nom)
                if self.i == len(self.noms) - 1:
                    print("Derniere image atteinte.")
                    break
                self.charger(self.i + 1)
            elif k == ord("a") and self.i > 0:
                self.donnees.valider(self.nom)
                self.charger(self.i - 1)
            elif k == ord("z"):
                self.annuler()
            elif k in (ord("+"), ord("=")):
                self.rayon = int(self.rayon * 1.1) + 1
                self.rayon_manuel = True
                self.dirty = True
            elif k in (ord("-"), ord("_")):
                self.rayon = max(5, int(self.rayon / 1.1))
                self.rayon_manuel = True
                self.dirty = True
            elif k in (ord("q"), 27):
                break

        self.donnees.sauver()
        cv2.destroyAllWindows()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.resume()

    def resume(self):
        total = 0
        ajouts = 0
        for nom in self.donnees.valides:
            for l in self.donnees.data.get(nom, []):
                if l["garde_manuel"]:
                    total += 1
                    if l["ajout_manuel"]:
                        ajouts += 1
        print("Termine :", len(self.donnees.valides), "images validees,",
              total, "cuvettes,", ajouts, "ajoutees a la main")
        print("CSV :", config.CSV_CORRIGE)
        print("Images :", config.DOSSIER_IMAGES_CORRIGEES)