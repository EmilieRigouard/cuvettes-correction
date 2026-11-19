# -*- coding: utf-8 -*-
"""
Georeferencement des cuvettes a partir du CSV de correction manuelle.

Plus besoin de georeferencer des images ni de chercher des pixels
rouges : on prend directement les cercles du CSV (centre + rayon en
pixels sur la photo d'origine) et on projette chaque cercle sur le MNT.

Etapes :
  1. Lecture du CSV (cuvettes gardees, images validees uniquement)
  2. Pour chaque photo : EXIF -> position/orientation de la camera
  3. Cercle (pixels) -> polygone au sol (rayons camera / MNT)
  4. Dedoublonnage : si deux cuvettes de photos differentes se
     touchent ou sont a moins de 40 cm, on n'en garde qu'une
  5. Export GeoPackage

Variables du .env (en plus de celles du script de correction) :
  DEM_PATH        chemin du MNT (GeoTIFF)
  EXIFTOOL_PATH   chemin de exiftool.exe
"""

import csv
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon

load_dotenv()


def chemin(nom):
    valeur = os.getenv(nom, "").strip().strip('"').strip("'")
    if not valeur:
        sys.exit("Variable " + nom + " manquante dans le .env")
    return Path(valeur)


# ═════════════════════════════════════════════════════════
# CHEMINS (.env)
# ═════════════════════════════════════════════════════════
DOSSIER_IMG = chemin("DOSSIER_IMAGES")      # photos drone d'origine
RESULTATS = chemin("DOSSIER_RESULTATS")     # dossier du CSV corrige
DEM_PATH = chemin("DEM_PATH")
EXIFTOOL_PATH = chemin("EXIFTOOL_PATH")

CSV_CUVETTES = RESULTATS / "detections_corrigees.csv"
FICHIER_VALIDES = RESULTATS / "images_validees.txt"
OUTPUT_GPKG = RESULTATS / "cuvettes.gpkg"

# ═════════════════════════════════════════════════════════
# PARAMETRES
# ═════════════════════════════════════════════════════════
EPSG_CODE = 32738
SENSOR_WIDTH_MM = 13.2
SENSOR_HEIGHT_MM = 8.8
FOCAL_LENGTH_MM = 8.8

NB_POINTS_CERCLE = 12      # points projetes par cercle (forme au sol)
MARGE_DEM_M = 300          # zone de MNT chargee autour du drone
MAX_PX_DEM = 4000          # MNT sous-echantillonne si plus grand
PAS_M = 0.5                # pas de la marche le long du rayon
PORTEE_MAX_M = 1500

# True = meme convention que ton ancien script (image retournee
# verticalement avant le georeferencement)
INVERSER_LIGNES = True

# Dedoublonnage entre photos DIFFERENTES : on garde une seule cuvette
# si elles se touchent ou si l'ecart entre leurs bords est inferieur
# ou egal a cette distance (en metres)
DISTANCE_MIN_M = 0.4

MARGE_BORD_PX = 5          # cercle colle au bord de la photo
INCLINAISON_ALERTE = 10    # degres d'ecart a la verticale avant alerte


# ═════════════════════════════════════════════════════════
# CAMERA : EXIF, distorsion, rayons
# ═════════════════════════════════════════════════════════
class Camera:
    def __init__(self, photo):
        self.photo = Path(photo)
        self.k1 = self.k2 = self.k3 = 0.0
        self.p1 = self.p2 = 0.0
        self.fx = self.fy = self.cx = self.cy = None
        self.corriger = False

    def lire_metadonnees(self):
        raw = subprocess.check_output([
            str(EXIFTOOL_PATH), "-json", "-G", "-n",
            "-EXIF:all", "-XMP:all", "-MakerNotes:all",
            "-Composite:all", str(self.photo)])
        meta = json.loads(raw)[0]

        def get(*keys):
            for k in keys:
                if k in meta:
                    return meta[k]
            return None

        def getf(*keys, default=0.0):
            for k in keys:
                if k in meta and meta[k] not in [None, ""]:
                    try:
                        return float(meta[k])
                    except Exception:
                        pass
            return default

        self.lat = get("Composite:GPSLatitude")
        self.lon = get("Composite:GPSLongitude")
        if self.lat is None or self.lon is None:
            raise ValueError("Pas de GPS dans l'EXIF")
        self.alt_abs = getf("MakerNotes:AbsoluteAltitude",
                            "XMP:AbsoluteAltitude")

        self.yaw_g = math.radians(getf("XMP:GimbalYawDegree"))
        self.pitch_g = math.radians(getf("XMP:GimbalPitchDegree"))
        self.roll_g = math.radians(getf("XMP:GimbalRollDegree"))
        self.dewarpflag = getf("XMP:DewarpFlag")

        dewarp = meta.get("XMP:DewarpData")
        if dewarp:
            try:
                motif = r"[-+]?\d*\.\d+|\d+"
                nums = list(map(float, re.findall(motif, dewarp)))
                if len(nums) > 9:
                    nums = nums[-9:]
                (self.fx, self.fy, _, _, self.k1, self.k2,
                 self.p1, self.p2, self.k3) = nums
            except Exception as e:
                print("  [WARN] DewarpData illisible :", e)

        self.cx = getf("XMP:CalibratedOpticalCenterX")
        self.cy = getf("XMP:CalibratedOpticalCenterY")

    def preparer(self, w, h):
        """Matrices de la camera (memes regles que l'ancien script)."""
        self.w = w
        self.h = h
        fx_n = FOCAL_LENGTH_MM * w / SENSOR_WIDTH_MM
        fy_n = FOCAL_LENGTH_MM * h / SENSOR_HEIGHT_MM
        K_nominal = np.array([[fx_n, 0, w / 2.0],
                              [0, fy_n, h / 2.0],
                              [0, 0, 1]], dtype=np.float64)

        if self.dewarpflag == 0:
            if self.fx and self.fy and self.cx and self.cy:
                K = np.array([[self.fx, 0, self.cx],
                              [0, self.fy, self.cy],
                              [0, 0, 1]], dtype=np.float64)
            else:
                K = K_nominal
            D = np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                         dtype=np.float64)
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
            self.K = K
            self.D = D
            self.new_K = new_K
            self.corriger = True
        else:
            self.new_K = K_nominal
            self.corriger = False

    @staticmethod
    def rotation(yaw, pitch, roll):
        pitch += math.radians(90)
        Rz = np.array([[math.cos(yaw), math.sin(yaw), 0],
                       [-math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
        return Rz @ Ry @ Rx

    def position(self):
        t = Transformer.from_crs("EPSG:4326", "EPSG:" + str(EPSG_CODE),
                                 always_xy=True)
        x, y = t.transform(self.lon, self.lat)
        return x, y, self.alt_abs

    def pixels_vers_modele(self, pts):
        """pts (N,2) : x = colonne, y = ligne sur la photo d'origine."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.corriger:
            pts = cv2.undistortPoints(pts, self.K, self.D, P=self.new_K)
        pts = pts.reshape(-1, 2).copy()
        if INVERSER_LIGNES:
            pts[:, 1] = (self.h - 1) - pts[:, 1]
        return pts

    def rayons(self, pts_modele):
        K_inv = np.linalg.inv(self.new_K)
        un = np.ones((len(pts_modele), 1))
        homog = np.hstack([pts_modele, un])
        rays_cam = homog @ K_inv.T
        R = self.rotation(self.yaw_g, self.pitch_g, self.roll_g)
        rays = rays_cam @ R.T
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays


# ═════════════════════════════════════════════════════════
# MNT : chargement local + intersection des rayons
# ═════════════════════════════════════════════════════════
class Mnt:
    def __init__(self, path):
        self.src = rasterio.open(path)
        self.meme_crs = self.src.crs == CRS.from_epsg(EPSG_CODE)
        self.vers_mnt = None
        if not self.meme_crs:
            self.vers_mnt = Transformer.from_crs(
                "EPSG:" + str(EPSG_CODE), self.src.crs, always_xy=True)

    def charger(self, cx, cy):
        """Charge la zone autour du drone. False si hors du MNT."""
        m = MARGE_DEM_M
        xs = [cx - m, cx + m, cx + m, cx - m]
        ys = [cy - m, cy - m, cy + m, cy + m]
        if not self.meme_crs:
            xs, ys = self.vers_mnt.transform(xs, ys)
        win = from_bounds(min(xs), min(ys), max(xs), max(ys),
                          self.src.transform)
        c0 = max(int(math.floor(win.col_off)), 0)
        r0 = max(int(math.floor(win.row_off)), 0)
        c1 = min(int(math.ceil(win.col_off + win.width)), self.src.width)
        r1 = min(int(math.ceil(win.row_off + win.height)),
                 self.src.height)
        if c1 <= c0 or r1 <= r0:
            return False
        win = Window(c0, r0, c1 - c0, r1 - r0)

        facteur = max(1.0, win.width / MAX_PX_DEM, win.height / MAX_PX_DEM)
        ow = max(1, int(win.width / facteur))
        oh = max(1, int(win.height / facteur))
        arr = self.src.read(1, window=win, out_shape=(oh, ow),
                            resampling=Resampling.bilinear, masked=True)
        arr = arr.astype("float32").filled(np.nan)
        self.arr = arr
        self.T = self.src.window_transform(win) * Affine.scale(
            win.width / ow, win.height / oh)
        return bool(np.isfinite(arr).any())

    def altitude(self, x, y):
        """Altitude du sol (interpolation bilinaire), NaN hors MNT."""
        if not self.meme_crs:
            xd, yd = self.vers_mnt.transform(x.ravel(), y.ravel())
            x = np.asarray(xd).reshape(x.shape)
            y = np.asarray(yd).reshape(y.shape)
        H, W = self.arr.shape
        colf = (x - self.T.c) / self.T.a - 0.5
        rowf = (y - self.T.f) / self.T.e - 0.5
        c0 = np.floor(colf).astype(int)
        r0 = np.floor(rowf).astype(int)
        fc = colf - c0
        fr = rowf - r0
        ok = (c0 >= 0) & (c0 < W - 1) & (r0 >= 0) & (r0 < H - 1)
        c0 = np.clip(c0, 0, W - 2)
        r0 = np.clip(r0, 0, H - 2)
        z = (self.arr[r0, c0] * (1 - fc) * (1 - fr)
             + self.arr[r0, c0 + 1] * fc * (1 - fr)
             + self.arr[r0 + 1, c0] * (1 - fc) * fr
             + self.arr[r0 + 1, c0 + 1] * fc * fr)
        z[~ok] = np.nan
        return z

    def intersecter(self, origine, rays):
        """Point d'impact (x, y) de chaque rayon sur le sol."""
        cx, cy, cz = origine
        n = len(rays)
        res = np.full((n, 2), np.nan)
        zmin = np.nanmin(self.arr)
        if cz <= zmin:
            return res

        for debut in range(0, n, 400):
            r = rays[debut:debut + 400]
            descend = r[:, 2] > 0.02
            if not descend.any():
                continue
            portee = (cz - zmin) / r[descend, 2].min() + 10
            portee = max(10.0, min(PORTEE_MAX_M, portee))
            t = np.arange(0.0, portee, PAS_M)

            x = cx + r[:, 0:1] * t[None, :]
            y = cy + r[:, 1:2] * t[None, :]
            z = cz - r[:, 2:3] * t[None, :]
            diff = z - self.altitude(x, y)

            neg = (diff <= 0) & descend[:, None]
            idx = neg.argmax(axis=1)
            ok = neg.any(axis=1) & (idx > 0)
            lignes = np.where(ok)[0]
            if len(lignes) == 0:
                continue

            i1 = idx[lignes]
            i0 = i1 - 1
            d0 = diff[lignes, i0]
            d1 = diff[lignes, i1]
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(np.isfinite(d0) & (d0 != d1),
                             d0 / (d0 - d1), 1.0)
            w = np.clip(w, 0.0, 1.0)
            th = t[i0] + w * (t[i1] - t[i0])
            res[debut + lignes, 0] = cx + th * r[lignes, 0]
            res[debut + lignes, 1] = cy + th * r[lignes, 1]
        return res


# ═════════════════════════════════════════════════════════
# LECTURE DU CSV
# ═════════════════════════════════════════════════════════
def lire_cuvettes():
    if not CSV_CUVETTES.exists():
        sys.exit("CSV introuvable : " + str(CSV_CUVETTES))
    if not FICHIER_VALIDES.exists():
        sys.exit("images_validees.txt introuvable : "
                 + str(FICHIER_VALIDES))

    texte = FICHIER_VALIDES.read_text(encoding="utf-8")
    valides = set(x.strip() for x in texte.splitlines() if x.strip())

    par_image = {}
    with open(CSV_CUVETTES, newline="", encoding="utf-8-sig") as f:
        for l in csv.DictReader(f, delimiter=";"):
            if int(l.get("garde_manuel") or 0) != 1:
                continue
            if l["image"] not in valides:
                continue
            par_image.setdefault(l["image"], []).append(l)

    print(len(valides), "images validees,",
          sum(len(v) for v in par_image.values()), "cuvettes gardees.")
    return par_image


# ═════════════════════════════════════════════════════════
# PROJECTION D'UNE PHOTO
# ═════════════════════════════════════════════════════════
def traiter_image(nom, cuvettes, mnt):
    photo = DOSSIER_IMG / nom
    img = cv2.imread(str(photo))
    if img is None:
        raise ValueError("Photo illisible : " + str(photo))
    h, w = img.shape[:2]

    cam = Camera(photo)
    cam.lire_metadonnees()
    cam.preparer(w, h)
    origine = cam.position()
    if not mnt.charger(origine[0], origine[1]):
        raise ValueError("Photo hors du MNT")
    incl = abs(math.degrees(cam.pitch_g) + 90.0)
    if incl > INCLINAISON_ALERTE:
        print("\n  [ATTENTION] camera inclinee de %.0f deg : modele "
              "valide pour des photos a la verticale." % incl)

    n_pts = NB_POINTS_CERCLE
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    infos = []
    tous = []
    for c in cuvettes:
        x1, y1 = float(c["x1"]), float(c["y1"])
        x2, y2 = float(c["x2"]), float(c["y2"])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        r = max(x2 - x1, y2 - y1) / 2.0
        tous.append([cx, cy])
        for a in angles:
            tous.append([cx + r * np.cos(a), cy + r * np.sin(a)])
        m = MARGE_BORD_PX
        bord = (x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m)
        dist = math.hypot(cx - w / 2.0, cy - h / 2.0)
        dist /= math.hypot(w / 2.0, h / 2.0)
        infos.append((c, bord, dist))

    pts = cam.pixels_vers_modele(np.array(tous))
    rays = cam.rayons(pts)
    xy = mnt.intersecter(origine, rays)
    xy = xy.reshape(len(cuvettes), n_pts + 1, 2)

    records = []
    for k, (c, bord, dist) in enumerate(infos):
        bloc = xy[k]
        if np.isnan(bloc).any():
            continue
        centre = bloc[0]
        anneau = bloc[1:]
        rayon_m = float(np.mean(np.hypot(anneau[:, 0] - centre[0],
                                         anneau[:, 1] - centre[1])))
        poly = Polygon(anneau)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        score = c.get("score", "")
        records.append({
            "image": nom,
            "id_csv": int(float(c["id"])),
            "ajout_manuel": int(c.get("ajout_manuel") or 0),
            "score": float(score) if score not in ("", None) else None,
            "diametre_m": round(2 * rayon_m, 3),
            "dist_centre": round(dist, 3),
            "touche_bord": int(bord),
            "inclinaison_deg": round(incl, 1),
            "geometry": poly,
        })
    return records


# ═════════════════════════════════════════════════════════
# DEDOUBLONNAGE ENTRE PHOTOS DIFFERENTES
# ═════════════════════════════════════════════════════════
def dedoublonner(gdf):
    """Si deux cuvettes de photos DIFFERENTES se touchent ou sont a
    moins de DISTANCE_MIN_M l'une de l'autre, on garde celle qui est
    la plus proche du centre de sa photo."""
    gdf = gdf.sort_values("dist_centre").reset_index(drop=True)
    geoms = list(gdf.geometry.values)
    images = gdf["image"].tolist()
    ids = gdf["id_csv"].tolist()
    sindex = gdf.sindex

    garde = np.ones(len(gdf), dtype=bool)
    doublon_de = [""] * len(gdf)

    for i in range(len(gdf)):
        zone = geoms[i].buffer(DISTANCE_MIN_M)
        for j in sindex.query(zone, predicate="intersects"):
            if j >= i or not garde[j] or images[j] == images[i]:
                continue
            if geoms[i].distance(geoms[j]) <= DISTANCE_MIN_M:
                garde[i] = False
                doublon_de[i] = images[j] + ":" + str(ids[j])
                break

    gdf["doublon_de"] = doublon_de
    return gdf[garde].copy(), gdf[~garde].copy()


# ═════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ═════════════════════════════════════════════════════════
def main():
    par_image = lire_cuvettes()
    if not par_image:
        sys.exit("Aucune cuvette a traiter.")

    mnt = Mnt(DEM_PATH)
    records = []
    echecs = 0
    noms = sorted(par_image)
    for k, nom in enumerate(noms, 1):
        print("[%d/%d] %s ..." % (k, len(noms), nom), end=" ", flush=True)
        try:
            recs = traiter_image(nom, par_image[nom], mnt)
            perdues = len(par_image[nom]) - len(recs)
            print(len(recs), "projetees", "(%d perdues)" % perdues
                  if perdues else "")
            records.extend(recs)
        except Exception as e:
            echecs += 1
            print("ECHEC :", e)
            traceback.print_exc()

    if not records:
        sys.exit("Aucune cuvette projetee.")

    crs = "EPSG:" + str(EPSG_CODE)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    avant = len(gdf)
    gardes, rejets = dedoublonner(gdf)
    print("\nDedoublonnage : %d -> %d (%d supprimees)" % (
        avant, len(gardes), len(rejets)))

    if OUTPUT_GPKG.exists():
        try:
            OUTPUT_GPKG.unlink()
        except PermissionError:
            sys.exit("Ferme QGIS : " + str(OUTPUT_GPKG) + " est ouvert.")
    gardes.to_file(OUTPUT_GPKG, layer="cuvettes", driver="GPKG")
    if len(rejets):
        rejets.to_file(OUTPUT_GPKG, layer="doublons_supprimes",
                       driver="GPKG")

    print("\nExport :", OUTPUT_GPKG)
    print("Couche cuvettes :", len(gardes), "entites")
    print("Photos en echec :", echecs)


if __name__ == "__main__":
    main()