import nibabel as nib
import numpy as np
from scipy import ndimage
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# =========================================================
# CONFIG
# =========================================================
base_dir = Path("/Users/chiki/Desktop/code/data")
output_dir = Path("/Users/chiki/Desktop/code/predictions")
output_dir.mkdir(exist_ok=True)

threshold = 2000
min_size = 12
max_size = 150

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

    labeled, _ = ndimage.label(binary)
    component_sizes = np.bincount(labeled.ravel())

    centers_world = []
    features = []

    for label_id, size in enumerate(component_sizes):

        if label_id == 0:
            continue

        # Filtre taille uniquement
        if not (min_size <= size <= max_size):
            continue

        mask = labeled == label_id
        coords = np.column_stack(np.where(mask))
        center_voxel = coords.mean(axis=0)

        # Conversion voxel → monde (LPS natif du CT)
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

    return np.array(centers_world), np.array(features), affine


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
# CLUSTERING ELECTRODES
# =========================================================
def cluster_electrodes(points):

    if len(points) == 0:
        return []

    clustering = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    ).fit(points)

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


def _to_slicer_ras(point_world, affine):
    """
    Slicer Markups (CoordinateSystem=0) attend du RAS.
    Nibabel est généralement en RAS+, mais certaines conversions peuvent produire du LPS.
    On détecte l'orientation et on flip X/Y uniquement si l'affine indique LPS.
    """
    try:
        ax = nib.aff2axcodes(affine)
    except Exception:
        ax = None

    x, y, z = float(point_world[0]), float(point_world[1]), float(point_world[2])

    # Si on est en LPS, convertir vers RAS (flip X et Y).
    if ax is not None and len(ax) >= 2 and ax[0] == "L" and ax[1] == "P":
        return (-x, -y, z)

    # Sinon, supposer déjà en RAS.
    return (x, y, z)


def export_markups_fcsv(points_world, out_path, affine):
    out_path = Path(out_path)
    with open(out_path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write("# CoordinateSystem = 0\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc\n")

        for i, p in enumerate(points_world):
            x_ras, y_ras, z_ras = _to_slicer_ras(p, affine)
            f.write(f"{i},{x_ras},{y_ras},{z_ras},0,0,0,1,1,1,0,Contact_{i},\n")

    return out_path


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("pat_")])
    all_data = []

    for patient_dir in patient_dirs:
        ct_path, xml_path = find_patient_files(patient_dir)
        if ct_path is None or xml_path is None:
            continue

        gt_points = read_ground_truth(xml_path)
        centers, X, affine = extract_candidates_and_features(ct_path)
        if len(X) == 0:
            continue

        y = label_candidates(centers, gt_points)

        all_data.append(
            {
                "id": patient_dir.name,
                "ct_path": ct_path,
                "xml_path": xml_path,
                "affine": affine,
                "gt_points": gt_points,
                "centers": centers,
                "X": X,
                "y": y,
            }
        )

    return all_data


def train_model(data_dir=base_dir):
    all_data = build_dataset(data_dir)
    if len(all_data) < 3:
        raise ValueError("Pas assez de patients utilisables pour entrainer.")

    X_train = np.vstack([p["X"] for p in all_data])
    y_train = np.hstack([p["y"] for p in all_data])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def predict_patient(patient_dir, model):
    patient_dir = Path(patient_dir)
    ct_path, _xml_path = find_patient_files(patient_dir)
    if ct_path is None:
        return {}, None

    centers, X, affine = extract_candidates_and_features(ct_path)
    if len(X) == 0:
        return {}, affine

    proba = model.predict_proba(X)[:, 1]
    pred = proba >= ml_probability_threshold
    kept = centers[pred]

    electrodes = cluster_electrodes(kept)
    result = {}
    for i, elec in enumerate(electrodes):
        result[f"E{i+1}"] = {"contacts": elec}

    return result, affine


if __name__ == "__main__":
    print("\n===== ML PIPELINE (mode script) =====\n")

    all_data = build_dataset(base_dir)
    print(f"Patients utilisables : {len(all_data)}")

    if len(all_data) < 3:
        raise SystemExit("Pas assez de patients utilisables pour une démo.")

    # Split léger pour debug local
    rng = np.random.default_rng(42)
    rng.shuffle(all_data)
    n_train = int(0.8 * len(all_data))
    train = all_data[:n_train]
    test = all_data[n_train:]

    X_train = np.vstack([p["X"] for p in train])
    y_train = np.hstack([p["y"] for p in train])
    X_test = np.vstack([p["X"] for p in test])
    y_test = np.hstack([p["y"] for p in test])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = y_proba >= ml_probability_threshold
    print("Matrice confusion")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report")
    print(classification_report(y_test, y_pred, digits=3))

    summary_csv = output_dir / "ml_slicer_export_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "gt_contacts", "pred_contacts", "fcsv_path"])

        for patient in test:
            gt_contacts = int(len(patient["gt_points"]))

            pred, affine = predict_patient(patient["ct_path"].parent, model)
            contacts = []
            for v in pred.values():
                contacts.extend(v["contacts"])
            contacts = np.asarray(contacts)

            out_fcsv = None
            if len(contacts) > 0:
                out_fcsv = output_dir / f"{patient['id']}_predicted_contacts.fcsv"
                export_markups_fcsv(contacts, out_fcsv, affine)

            print(f"{patient['id']}: GT={gt_contacts}, pred={len(contacts)}")
            w.writerow([patient["id"], gt_contacts, len(contacts), str(out_fcsv) if out_fcsv else ""])

    print("\nRésumé écrit :", summary_csv)
