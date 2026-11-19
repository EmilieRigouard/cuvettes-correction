"""Zoom et deplacement dans la fenetre de correction."""

import math

import cv2
import numpy as np


class Vue:
    """Gere la partie de l'image visible a l'ecran.

    Coordonnees "image" : pixels de la photo d'origine.
    Coordonnees "ecran" : pixels de la fenetre.
    """

    def __init__(self, largeur, hauteur, max_w, max_h, zoom_max, pas):
        self.w = largeur
        self.h = hauteur
        self.base = min(max_w / largeur, max_h / hauteur, 1.0)
        self.vw = max(1, int(round(largeur * self.base)))
        self.vh = max(1, int(round(hauteur * self.base)))
        self.zoom_max = zoom_max
        self.pas = pas
        self.zoom = 1.0
        self.x0 = 0.0     # coin haut-gauche visible (coord. image)
        self.y0 = 0.0

    @property
    def echelle(self):
        """Pixels d'ecran par pixel d'image."""
        return self.base * self.zoom

    def vers_image(self, sx, sy):
        s = self.echelle
        return self.x0 + sx / s, self.y0 + sy / s

    def vers_ecran(self, x, y):
        s = self.echelle
        return (x - self.x0) * s, (y - self.y0) * s

    def _limiter(self):
        s = self.echelle
        max_x = max(0.0, self.w - self.vw / s)
        max_y = max(0.0, self.h - self.vh / s)
        self.x0 = min(max(self.x0, 0.0), max_x)
        self.y0 = min(max(self.y0, 0.0), max_y)

    def reinitialiser(self):
        self.zoom = 1.0
        self.x0 = 0.0
        self.y0 = 0.0

    def zoomer(self, facteur, sx=None, sy=None):
        """Zoom centre sur le point ecran (sx, sy), ou sur le centre."""
        if sx is None:
            sx = self.vw / 2.0
        if sy is None:
            sy = self.vh / 2.0
        px, py = self.vers_image(sx, sy)
        self.zoom = min(max(self.zoom * facteur, 1.0), self.zoom_max)
        s = self.echelle
        self.x0 = px - sx / s
        self.y0 = py - sy / s
        self._limiter()

    def deplacer(self, dsx, dsy):
        """Deplace la zone visible de (dsx, dsy) pixels ECRAN."""
        s = self.echelle
        self.x0 += dsx / s
        self.y0 += dsy / s
        self._limiter()

    def rendre(self, img):
        """Image affichee (taille fixe vw x vh) pour l'etat actuel."""
        s = self.echelle
        xi = int(math.floor(self.x0))
        yi = int(math.floor(self.y0))
        cw = min(self.w - xi, int(math.ceil(self.vw / s)) + 2)
        ch = min(self.h - yi, int(math.ceil(self.vh / s)) + 2)
        crop = img[yi:yi + ch, xi:xi + cw]
        nw = max(1, int(round(cw * s)))
        nh = max(1, int(round(ch * s)))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        grand = cv2.resize(crop, (nw, nh), interpolation=interp)
        fx = int(round((self.x0 - xi) * s))
        fy = int(round((self.y0 - yi) * s))
        morceau = grand[fy:fy + self.vh, fx:fx + self.vw]
        canvas = np.zeros((self.vh, self.vw, 3), dtype=np.uint8)
        canvas[:morceau.shape[0], :morceau.shape[1]] = morceau
        return canvas