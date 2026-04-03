import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# CHEMIN DATA
# =========================
base_dir = Path("/Users/chiki/Desktop/code/data")


# =========================
# LECTURE GROUND TRUTH XML
# =========================
def read_gt_xml(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    electrodes = []

    for electrode in root.findall("Electrode"):
        plots = electrode.find("Plots")
        if plots is not None:
            points = []
            for plot in plots.findall("Plot"):
                x = float(plot.attrib["x"])
                y = float(plot.attrib["y"])
                z = float(plot.attrib["z"])
                points.append([x, y, z])

            electrodes.append(np.array(points))

    return electrodes


# =========================
# LECTURE PRED FCSV
# =========================
def read_fcsv(fcsv_path):

    points = []

    with open(fcsv_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split(",")
            if len(parts) >= 4:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                points.append([x, y, z])

    return np.array(points)


# =========================
# CLUSTERING PRED EN ELECTRODES
# =========================
def associate_contacts(points,
                       eps=6,
                       min_samples=3,
                       linearity_threshold=0.9):

    if len(points) == 0:
        return []

    clustering = DBSCAN(eps=eps,
                        min_samples=min_samples).fit(points)

    labels = clustering.labels_
    electrodes = []

    for label in np.unique(labels):

        if label == -1:
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < 4:
            continue

        pca = PCA(n_components=3)
        pca.fit(cluster_points)

        # Test de linéarité
        if pca.explained_variance_ratio_[0] < linearity_threshold:
            continue

        projection = pca.transform(cluster_points)[:, 0]
        order = np.argsort(projection)
        ordered = cluster_points[order]

        electrodes.append(ordered)

    return electrodes


# =========================
# VISUALISATION MULTI-PATIENTS
# =========================
print("\nVisualisation GT vs PRED pour tous les patients\n")

for patient_dir in sorted(base_dir.glob("pat_*")):

    patient_id = patient_dir.name

    xml_files = list(patient_dir.glob("*electrodes*.xml"))
    fcsv_file = base_dir / f"{patient_id}_predicted_contacts.fcsv"

    if not xml_files or not fcsv_file.exists():
        continue

    print(f"\nPatient : {patient_id}")

    gt_electrodes = read_gt_xml(xml_files[0])
    pred_points = read_fcsv(fcsv_file)
    print("Nb pred points:", len(pred_points))
    print("Min coords:", np.min(pred_points, axis=0))
    print("Max coords:", np.max(pred_points, axis=0))
    pred_electrodes = associate_contacts(pred_points)

    # Convertir toutes les électrodes GT en un seul tableau
    if len(gt_electrodes) > 0:
        gt_points = np.vstack(gt_electrodes)
    else:
        gt_points = np.array([])

    print(f"GT electrodes : {len(gt_electrodes)}")
    print(f"Pred electrodes : {len(pred_electrodes)}")

    # ----- Vérification RAS / LPS -----
    if len(gt_points) > 0 and len(pred_points) > 0:
        print("Centre GT:", np.mean(gt_points, axis=0))
        print("Centre Pred:", np.mean(pred_points, axis=0))

    # =========================
    # PLOT
    # =========================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ----- GT -----
    first = True
    for elec in gt_electrodes:
        if first:
            ax.plot(elec[:,0], elec[:,1], elec[:,2],
                    linewidth=1,
                    marker='o',
                    label="Ground Truth")
            first = False
        else:
            ax.plot(elec[:,0], elec[:,1], elec[:,2],
                    linewidth=1,
                    marker='o')

    # ----- PRED -----
    first = True
    for elec in pred_electrodes:
        if first:
            ax.plot(elec[:,0], elec[:,1], elec[:,2],
                    linewidth=3,
                    marker='^',
                    label="Prediction")
            first = False
        else:
            ax.plot(elec[:,0], elec[:,1], elec[:,2],
                    linewidth=3,
                    marker='^')

    ax.set_title(f"GT vs PRED - {patient_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

print("\nTerminé.")