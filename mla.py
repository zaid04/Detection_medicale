import nibabel as nib
import numpy as np
from scipy import ndimage
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# =========================================================
# CONFIG
# =========================================================
base_dir = Path("/Users/chiki/Desktop/code/data")

threshold = 2000
min_size = 12
max_size = 150

# ❌ ROI SUPPRIMÉ (problème multi-patients)
# x_min, x_max = 80, 430
# y_min, y_max = 80, 430
# z_min, z_max = 20, 250

match_threshold = 8.0
ml_probability_threshold = 0.15
dbscan_eps = 7
dbscan_min_samples = 3


# =========================================================
# LECTURE XML
# =========================================================
def read_ground_truth(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_points = []

    for electrode in root.findall("Electrode"):
        plots = electrode.find("Plots")
        if plots is not None:
            for plot in plots.findall("Plot"):
                x = float(plot.attrib.get("x", 0))
                y = float(plot.attrib.get("y", 0))
                z = float(plot.attrib.get("z", 0))
                gt_points.append([x, y, z])

    return np.array(gt_points)


# =========================================================
# TROUVER FICHIERS
# =========================================================
def find_patient_files(patient_dir):
    ct_candidates = list(patient_dir.rglob("*ct_post*.nii.gz"))
    xml_candidates = list(patient_dir.rglob("*electrodes*.xml"))

    if not ct_candidates or not xml_candidates:
        return None, None

    return ct_candidates[0], xml_candidates[0]


# =========================================================
# EXTRACTION CANDIDATS
# =========================================================
def extract_candidates_and_features(ct_path):

    ct = nib.load(str(ct_path))
    data = ct.get_fdata()
    affine = ct.affine

    # Seuillage métal
    binary = data > threshold

    # Composantes connectées
    labeled, _ = ndimage.label(binary)
    component_sizes = np.bincount(labeled.ravel())

    centers_world = []
    features = []

    for label_id, size in enumerate(component_sizes):

        if label_id == 0:
            continue

        # Filtre taille
        if not (min_size <= size <= max_size):
            continue

        mask = labeled == label_id
        coords = np.column_stack(np.where(mask))
        center_voxel = coords.mean(axis=0)

        # ❌ ROI voxel supprimé
        # On laisse tous les candidats passer

        # Conversion voxel -> monde
        center_world = nib.affines.apply_affine(affine, center_voxel)

        intensities = data[mask]

        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        dx, dy, dz = maxs - mins + 1

        feat = [
            size,
            center_world[0], center_world[1], center_world[2],
            np.mean(intensities),
            np.max(intensities),
            np.min(intensities),
            np.std(intensities),
            dx, dy, dz
        ]

        centers_world.append(center_world)
        features.append(feat)

    return np.array(centers_world), np.array(features)


# =========================================================
# LABELISATION
# =========================================================
def label_candidates(centers_world, gt_points):
    labels = []
    for c in centers_world:
        distances = np.linalg.norm(gt_points - c, axis=1)
        labels.append(1 if np.min(distances) < match_threshold else 0)
    return np.array(labels)


# =========================================================
# CLUSTERING
# =========================================================
def cluster_electrodes(points):

    if len(points) == 0:
        return []

    clustering = DBSCAN(eps=dbscan_eps,
                        min_samples=dbscan_min_samples).fit(points)

    labels = clustering.labels_
    electrodes = []

    for label in np.unique(labels):
        if label == -1:
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) > 1:
            pca = PCA(n_components=1)
            projected = pca.fit_transform(cluster_points).ravel()
            order = np.argsort(projected)
            cluster_points = cluster_points[order]

        electrodes.append(cluster_points)

    return electrodes


# =========================================================
# CONSTRUCTION DATASET
# =========================================================
patient_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
all_data = []

print("\nConstruction dataset...\n")

for patient_dir in patient_dirs:

    ct_path, xml_path = find_patient_files(patient_dir)
    if ct_path is None:
        continue

    gt_points = read_ground_truth(xml_path)
    centers, X = extract_candidates_and_features(ct_path)

    print(f"{patient_dir.name} -> candidats détectés :", len(centers))

    if len(X) == 0:
        continue

    y = label_candidates(centers, gt_points)

    all_data.append({
        "id": patient_dir.name,
        "gt_points": gt_points,
        "centers": centers,
        "X": X,
        "y": y
    })

print("Patients utilisables :", len(all_data))