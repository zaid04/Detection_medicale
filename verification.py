import numpy as np
import xml.etree.ElementTree as ET
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =====================================
# CHEMINS
# =====================================
xml_path = "/Users/chiki/Desktop/code/data/pat_03415_1893/pat_03415_1893_electrodes_withModels.xml"
fcsv_path = "/Users/chiki/Desktop/code/data/pat_03415_1893_predicted_contacts.fcsv"


# =====================================
# LECTURE GROUND TRUTH (LPS → RAS)
# =====================================
def read_ground_truth(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_points = []

    for electrode in root.findall("Electrode"):
        plots = electrode.find("Plots")
        if plots is not None:
            for plot in plots.findall("Plot"):

                x_lps = float(plot.attrib.get("x", 0))
                y_lps = float(plot.attrib.get("y", 0))
                z_lps = float(plot.attrib.get("z", 0))

                # Conversion LPS → RAS
                x_ras = -x_lps
                y_ras = -y_lps
                z_ras =  z_lps

                gt_points.append([x_ras, y_ras, z_ras])

    return np.array(gt_points)


# =====================================
# LECTURE FCSV
# =====================================
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


# =====================================
# ASSOCIATION CONTACTS → ELECTRODES
# =====================================
def associate_contacts_to_electrodes(points,
                                     eps=7,
                                     min_samples=3,
                                     linearity_threshold=0.85):

    clustering = DBSCAN(eps=eps,
                        min_samples=min_samples).fit(points)

    labels = clustering.labels_
    electrodes = []

    for label in np.unique(labels):

        if label == -1:
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < 3:
            continue

        # PCA pour tester la linéarité
        pca = PCA(n_components=3)
        pca.fit(cluster_points)

        variance_ratio = pca.explained_variance_ratio_

        # Vérifie que c’est bien une ligne
        if variance_ratio[0] > linearity_threshold:

            # Ordonner les points le long de la ligne
            projection = pca.transform(cluster_points)[:, 0]
            order = np.argsort(projection)
            ordered_points = cluster_points[order]

            electrodes.append(ordered_points)

    return electrodes


# =====================================
# VISUALISATION 3D
# =====================================
def plot_electrodes(electrodes):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0,1,len(electrodes)))

    for i, electrode in enumerate(electrodes):

        pts = np.array(electrode)

        ax.scatter(pts[:,0],
                   pts[:,1],
                   pts[:,2],
                   color=colors[i],
                   s=50,
                   label=f"Electrode {i}")

        ax.plot(pts[:,0],
                pts[:,1],
                pts[:,2],
                color=colors[i])

    ax.set_title("Electrodes détectées (Prédiction)")
    ax.legend()
    plt.show()


# =====================================
# MAIN
# =====================================
gt_points = read_ground_truth(xml_path)
pred_points = read_fcsv(fcsv_path)

print("GT contacts:", len(gt_points))
print("Pred contacts:", len(pred_points))

electrodes = associate_contacts_to_electrodes(pred_points)

print("\nNombre électrodes détectées :", len(electrodes))

for i, electrode in enumerate(electrodes):
    print(f"Electrode {i} : {len(electrode)} contacts")

plot_electrodes(electrodes)

print("\nTerminé.")