import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET
from itertools import product

# =========================
# 1. Lire le ground truth XML
# =========================
def read_ground_truth(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_num_electrodes = int(root.attrib.get("numElectrodes", 0))
    gt_num_contacts = 0

    for electrode in root.findall("Electrode"):
        plots = electrode.find("Plots")
        if plots is not None:
            gt_num_contacts += len(plots.findall("Plot"))

    return gt_num_electrodes, gt_num_contacts


# =========================
# 2. Pipeline de détection
# =========================
def run_pipeline(
    ct_path,
    threshold,
    min_size,
    max_size,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    dbscan_eps,
    dbscan_min_samples,
):
    ct = nib.load(ct_path)
    data = ct.get_fdata()

    # Seuillage
    binary = data > threshold

    # Composantes connectées 3D
    labeled, _ = ndimage.label(binary)

    # Taille des composantes
    component_sizes = np.bincount(labeled.ravel())

    valid_labels = []
    for label_id, size in enumerate(component_sizes):
        if label_id == 0:
            continue
        if min_size <= size <= max_size:
            valid_labels.append(label_id)

    # Filtrage spatial
    filtered_centers = []
    for label_id in valid_labels:
        coords = np.argwhere(labeled == label_id)
        center = coords.mean(axis=0)
        x, y, z = center

        if (
            x_min < x < x_max and
            y_min < y < y_max and
            z_min < z < z_max
        ):
            filtered_centers.append(center)

    filtered_centers = np.array(filtered_centers)

    # Clustering
    if len(filtered_centers) == 0:
        return {
            "num_centers": 0,
            "num_electrodes": 0,
        }

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(filtered_centers)

    unique_labels = set(labels)
    num_electrodes = len([l for l in unique_labels if l != -1])

    return {
        "num_centers": len(filtered_centers),
        "num_electrodes": num_electrodes,
    }


# =========================
# 3. Paramètres patient
# =========================
ct_path = "/Users/chiki/Desktop/patient/pat_03601_2081/nifti/pat_03601_2081_ct_post.nii.gz"
xml_path = "/Users/chiki/Downloads/DataHandsOn2/pat_03601_2081_electrodes_withModels.xml"

gt_num_electrodes, gt_num_contacts = read_ground_truth(xml_path)

print("Ground truth électrodes :", gt_num_electrodes)
print("Ground truth contacts   :", gt_num_contacts)

# =========================
# 4. Paramètres fixes
# =========================
x_min, x_max = 80, 430
y_min, y_max = 80, 430
z_min, z_max = 20, 250

# =========================
# 5. Grille de recherche (VERSION RAPIDE)
# =========================
threshold_values = [1800, 1900, 2000]
min_size_values = [8, 10, 12]
max_size_values = [120, 150]
dbscan_eps_values = [8, 10, 12]
dbscan_min_samples_values = [2]

results = []

total_tests = (
    len(threshold_values)
    * len(min_size_values)
    * len(max_size_values)
    * len(dbscan_eps_values)
    * len(dbscan_min_samples_values)
)
current_test = 1

for threshold, min_size, max_size, dbscan_eps, dbscan_min_samples in product(
    threshold_values,
    min_size_values,
    max_size_values,
    dbscan_eps_values,
    dbscan_min_samples_values,
):
    print(
        f"\nTest {current_test}/{total_tests} "
        f"-> threshold={threshold}, min_size={min_size}, "
        f"max_size={max_size}, eps={dbscan_eps}, min_samples={dbscan_min_samples}"
    )
    current_test += 1

    out = run_pipeline(
        ct_path=ct_path,
        threshold=threshold,
        min_size=min_size,
        max_size=max_size,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    num_centers = out["num_centers"]
    num_electrodes = out["num_electrodes"]

    error_electrodes = abs(num_electrodes - gt_num_electrodes)
    error_contacts = abs(num_centers - gt_num_contacts)

    score = error_electrodes * 10 + error_contacts

    results.append({
        "threshold": threshold,
        "min_size": min_size,
        "max_size": max_size,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "num_centers": num_centers,
        "num_electrodes": num_electrodes,
        "error_electrodes": error_electrodes,
        "error_contacts": error_contacts,
        "score": score,
    })

# =========================
# 6. Trier du meilleur au moins bon
# =========================
results = sorted(results, key=lambda x: x["score"])

print("\n=== Top réglages ===")
for r in results:
    print(
        f"threshold={r['threshold']}, "
        f"min_size={r['min_size']}, "
        f"max_size={r['max_size']}, "
        f"eps={r['dbscan_eps']}, "
        f"min_samples={r['dbscan_min_samples']} "
        f"-> centres={r['num_centers']}, "
        f"electrodes={r['num_electrodes']}, "
        f"err_elec={r['error_electrodes']}, "
        f"err_contacts={r['error_contacts']}, "
        f"score={r['score']}"
    )