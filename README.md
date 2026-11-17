# Cuvettes YOLO

Detection automatique de cuvettes de degazage (YOLO + SAHI), filtrage par
regles, puis correction manuelle image par image (suppression des faux
positifs et ajout des cuvettes oubliees).

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Configuration

1. Copier `.env.example` en `.env`
2. Renseigner les chemins : modele YOLO (`best.pt`), dossier d'images,
   dossier de resultats.

Les parametres de detection et de filtrage sont dans `cuvettes/config.py`.

## Utilisation

```bash
python main.py
```

| Action | Commande |
|---|---|
| Supprimer / remettre un rond | clic gauche |
| Ajouter une cuvette | clic droit (taille par defaut) |
| Ajouter en choisissant la taille | clic droit + glisser (centre vers bord) |
| Valider et image suivante | `d`, Espace ou Entree |
| Image precedente | `a` |
| Annuler la derniere action | `z` |
| Taille par defaut des ajouts | `+` / `-` |
| Sauvegarder et quitter | `q` ou Echap |

Les ronds rouges sont gardes, les ronds orange fins sont rejetes, les ronds
verts sont ajoutes a la main. Le travail est sauvegarde a chaque validation :
relancer le script reprend a la premiere image non validee.

## Sorties (dans le dossier de resultats)

- `detections_corrigees.csv` : toutes les detections (colonnes
  `garde_final` = decision automatique, `garde_manuel` = decision
  manuelle, `ajout_manuel` = 1 pour les ajouts)
- `images/` : images annotees finales
- `images_validees.txt` : liste des images deja validees

## Notes

- Les modeles (`.pt`), les images et le `.env` ne sont pas versionnes.
- Lancer le script dans un terminal (pas dans la console Spyder) pour que
  la fenetre OpenCV reagisse bien.