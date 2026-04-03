import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ==========================
# CONFIG
# ==========================
base_dir = Path("/Users/chiki/Desktop/code/data")
dbscan_eps = 7
dbscan_min_samples = 3


# ==========================
# LECTURE XML GT + CONVERSION RAS → LPS
# ==========================
def read_gt_structure(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    electrodes = []

    for electrode in root.findall("Electrode"):

        name = electrode.attrib.get("Name", "Unknown")
        plots = electrode.find("Plots")

        points = []

        if plots is not None:
            for plot in plots.findall("Plot"):
                x = float(plot.attrib["x"])
                y = float(plot.attrib["y"])
                z = float(plot.attrib["z"])
                points.append([x, y, z])

        pts = np.array(points)

        if len(pts) > 0:
            # 🔥 RAS → LPS
            pts[:, 0] *= -1
            pts[:, 1] *= -1

        electrodes.append({
            "name": name,
            "points": pts
        })

    return electrodes


# ==========================
# LECTURE FCSV PRED (déjà LPS)
# ==========================
def read_pred_fcsv(fcsv_path):

    points = []

    with open(fcsv_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split(",")

            if len(parts) < 4:
                continue

            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            points.append([x, y, z])

    return np.array(points)


# ==========================
# CLUSTERING PRED EN ELECTRODES
# ==========================
def cluster_pred(points):

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


# ==========================
# VISUALISATION MULTI-PATIENTS
# ==========================
print("\nVisualisation GT vs PRED\n")

for patient_dir in sorted(base_dir.glob("pat_*")):

    xml_files = list(patient_dir.glob("*electrodes*.xml"))
    pred_file = base_dir / f"{patient_dir.name}_predicted_contacts.fcsv"

    if not xml_files or not pred_file.exists():
        continue

    print("Patient :", patient_dir.name)

    gt_electrodes = read_gt_structure(xml_files[0])
    pred_points = read_pred_fcsv(pred_file)
    pred_electrodes = cluster_pred(pred_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- GT en bleu ----
    for i, elec in enumerate(gt_electrodes):

        pts = elec["points"]

        if len(pts) == 0:
            continue

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            marker='o',
            linewidth=1,
            color='blue',
            label="Ground Truth" if i == 0 else ""
        )

    # ---- PRED en rouge ----
    for i, elec in enumerate(pred_electrodes):

        if len(elec) == 0:
            continue

        ax.plot(
            elec[:, 0],
            elec[:, 1],
            elec[:, 2],
            marker='^',
            linewidth=3,
            color='red',
            label="Prediction" if i == 0 else ""
        )

    ax.set_title(f"GT vs PRED - {patient_dir.name}")
    ax.set_xlabel("X (LPS)")
    ax.set_ylabel("Y (LPS)")
    ax.set_zlabel("Z (LPS)")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 🔎 DEBUG CENTRES
    if len(gt_electrodes) > 0:
        all_gt = np.vstack([e["points"] for e in gt_electrodes if len(e["points"]) > 0])
        print("Centre GT (LPS):", np.mean(all_gt, axis=0))

    if len(pred_points) > 0:
        print("Centre PRED (LPS):", np.mean(pred_points, axis=0))

print("\nTerminé.")