# SEEG Electrode Localization – M2 IDMC 2026

## Description
Prototype de localisation automatique des contacts d’électrodes SEEG à partir d’un CT post-opératoire.

Le pipeline comprend :
- Détection des voxels métalliques (seuillage HU)
- Extraction de features
- Classification (Random Forest)
- Regroupement en électrodes (DBSCAN)
- Visualisation dans 3D Slicer

---

## Fichiers principaux

- `ml.py` : pipeline complet (entraînement + prédiction)
- `EpilocView.py` : module 3D Slicer pour lancer la localisation

---

## Utilisation

### Mode script
```bash
python ml.py

## Données

Les données complètes (20 patients) ne sont pas incluses dans cette archive afin d’alléger le rendu.
Le modèle a été entraîné et validé sur le corpus fourni dans le cadre du projet.