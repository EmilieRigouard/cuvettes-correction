# Cuvettes YOLO

Détection automatique de cuvettes de dégazage sur des photos de drone
(YOLO + SAHI), correction manuelle image par image, puis géoréférencement
des cuvettes validées (GeoPackage à ouvrir dans QGIS).

Le projet a deux étapes :

1. `main.py` : détection, filtrage par règles et correction manuelle
   (suppression des faux positifs, ajout des cuvettes oubliées, zoom).
2. `georef_csv.py` : projection des cuvettes validées sur le MNT, suppression
   des doublons entre photos et export en GeoPackage.

## Structure

```
main.py              étape 1 : détection + correction manuelle
georef_csv.py        étape 2 : géoréférencement
cuvettes/
    config.py        chemins (.env) et paramètres de détection / filtres
    detection.py     YOLO + SAHI, features, filtres par règles
    donnees.py       lecture / écriture du CSV et des images validées
    interface.py     fenêtre de correction
    vue.py           zoom et déplacement
```

## Installation

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si la fenêtre ne s'ouvre pas (erreur « The function is not implemented »),
c'est qu'une version « headless » d'OpenCV a été installée :

```powershell
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python
```

## Configuration

Copier `.env.example` en `.env` et renseigner les chemins (sans guillemets,
avec des `/`) :

| Variable | Rôle |
|---|---|
| `MODELE_YOLO` | modèle YOLO (`best.pt`) |
| `DOSSIER_IMAGES` | photos drone d'origine (`.jpg`) |
| `DOSSIER_RESULTATS` | dossier de sortie (CSV, images annotées, GeoPackage) |
| `MODELE_CLASSIFIEUR` | classifieur optionnel (laisser vide sinon) |
| `DEVICE` | `cpu` ou `cuda` |
| `DEM_PATH` | MNT (GeoTIFF), pour l'étape 2 |
| `EXIFTOOL_PATH` | chemin de `exiftool.exe`, pour l'étape 2 |

Les paramètres de détection et de filtrage sont dans `cuvettes/config.py`.
Le fichier `.env` n'est pas versionné.

## Étape 1 : détection et correction manuelle

```powershell
python main.py
```

Pour chaque image, YOLO détecte les cuvettes (la détection de l'image
suivante se fait en arrière-plan pendant la correction), puis la fenêtre
s'ouvre.

Code couleur : rond rouge = gardé, rond orange fin = supprimé,
rond vert = ajouté à la main.

| Action | Commande |
|---|---|
| Supprimer / remettre un rond | clic gauche |
| Ajouter une cuvette | clic droit (rayon par défaut : 20 px) |
| Ajouter en choisissant la taille | clic droit + glisser (centre vers bord) |
| Taille par défaut des ajouts | `+` / `-` |
| Valider et passer à l'image suivante | `d`, Espace ou Entrée |
| Image précédente | `a` |
| Annuler la dernière action | `z` |
| Zoom / dézoom | molette, deux doigts sur le pavé tactile, ou `e` / `r` |
| Vue entière | `0` |
| Se déplacer dans l'image zoomée | flèches, ou Maj + clic gauche + glisser |
| Sauvegarder et quitter | `q` ou Échap |

Le zoom revient à la vue entière à chaque nouvelle image.

Le travail est sauvegardé à chaque validation. Relancer `main.py` reprend à
la première image non validée.

### Sorties (dans `DOSSIER_RESULTATS`)

- `detections_corrigees.csv` : toutes les détections. `garde_final` est la
  décision automatique, `garde_manuel` la décision manuelle, `ajout_manuel`
  vaut 1 pour les cuvettes ajoutées à la main. La colonne `label` reprend
  `garde_manuel` (utile pour entraîner un classifieur).
- `images/` : images annotées finales (uniquement les ronds gardés).
- `images_validees.txt` : liste des images validées avec `d`.

## Étape 2 : géoréférencement

```powershell
python georef_csv.py
```

Le script lit `detections_corrigees.csv` et `images_validees.txt`. Seules
les cuvettes gardées (`garde_manuel = 1`) des images **validées** sont
traitées. Pour chaque photo, il lit les métadonnées EXIF (position et
orientation de la caméra), puis projette le cercle de chaque cuvette sur le
MNT : on obtient un polygone au sol.

**Doublons entre photos.** Si deux cuvettes de photos différentes se
touchent ou sont à 40 cm ou moins l'une de l'autre, une seule est gardée :
celle qui est la plus proche du centre de sa photo. La distance se règle
avec `DISTANCE_MIN_M` dans `georef_csv.py`. Deux cuvettes de la même photo
ne sont jamais fusionnées.

### Sortie : `cuvettes.gpkg` (dans `DOSSIER_RESULTATS`)

- Couche `cuvettes` : cuvettes conservées (EPSG:32738).
- Couche `doublons_supprimes` : cuvettes retirées, avec la colonne
  `doublon_de` qui indique laquelle a été gardée. À regarder dans QGIS pour
  vérifier le réglage de `DISTANCE_MIN_M`.

| Colonne | Contenu |
|---|---|
| `image` | photo d'origine |
| `id_csv` | identifiant dans `detections_corrigees.csv` |
| `ajout_manuel` | 1 si la cuvette a été ajoutée à la main |
| `score` | score YOLO (vide pour les ajouts manuels) |
| `diametre_m` | diamètre au sol, en mètres |
| `dist_centre` | distance au centre de la photo (0 = centre, 1 = coin) |
| `touche_bord` | 1 si le cercle touche le bord de la photo |
| `inclinaison_deg` | écart de la caméra par rapport à la verticale |

Fermer QGIS avant de relancer le script s'il a le GeoPackage ouvert.

### Limites

- Le modèle de caméra n'est validé que pour des photos prises à la
  verticale. Le script affiche `[ATTENTION]` si l'inclinaison dépasse 10°,
  et la colonne `inclinaison_deg` permet de filtrer ces photos.
- Si les cuvettes sont décalées ou en miroir par rapport à la réalité,
  essayer `INVERSER_LIGNES = False` dans `georef_csv.py`.
- Les paramètres du capteur (`SENSOR_WIDTH_MM`, `SENSOR_HEIGHT_MM`,
  `FOCAL_LENGTH_MM`) et l'EPSG (`EPSG_CODE`) sont en haut de
  `georef_csv.py`.

## Notes

- Les modèles (`.pt`), les images et le `.env` ne sont pas versionnés.
- Lancer les scripts dans un terminal (PowerShell), pas dans la console de
  Spyder, pour que la fenêtre OpenCV réagisse bien.