import numpy as np
import nibabel as nib
import csv
from pathlib import Path
from scipy.ndimage import label
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ======================================
# CONFIG
# ======================================
base_dir = Path("/Users/chiki/Desktop/code/data")
output_dir = Path("/Users/chiki/Desktop/code/predictions")
output_dir.mkdir(exist_ok=True)

threshold_hu = 2000
dbscan_eps = 7
dbscan_min_samples = 3


# ======================================
# DETECTION CONTACTS (CT NIFTI)
# ======================================
def detect_contacts(ct_path):

    img = nib.load(str(ct_path))
    data = img.get_fdata()

    # Seuillage métal
    mask = data > threshold_hu

    # Composantes connectées 3D
    labeled, num = label(mask)

    centers_voxel = []

    for i in range(1, num + 1):

        coords = np.argwhere(labeled == i)

        # Filtre taille réaliste contact
        if 10 < len(coords) < 150:
            center = coords.mean(axis=0)
            centers_voxel.append(center)

    if len(centers_voxel) == 0:
        return np.array([])

    centers_voxel = np.array(centers_voxel)

    # voxel → world
    centers_world = nib.affines.apply_affine(img.affine, centers_voxel)

    return centers_world


# ======================================
# REGROUPEMENT EN ELECTRODES
# ======================================
def cluster_electrodes(points):

    if len(points) == 0:
        return []

    clustering = DBSCAN(
        eps=dbscan_eps,
        min_samples=dbscan_min_samples
    ).fit(points)

    labels = clustering.labels_
    electrodes = []

    for label_id in np.unique(labels):

        if label_id == -1:
            continue

        cluster_points = points[labels == label_id]

        # Ordonner le long de l’axe principal
        if len(cluster_points) > 1:
            pca = PCA(n_components=1)
            proj = pca.fit_transform(cluster_points).ravel()
            order = np.argsort(proj)
            cluster_points = cluster_points[order]

        electrodes.append(cluster_points)

    return electrodes


# ======================================
# PIPELINE 20 PATIENTS
# ======================================
print("\n===== GENERATION CSV POUR 20 PATIENTS =====\n")

for patient_dir in sorted(base_dir.glob("pat_*")):

    print("Patient :", patient_dir.name)

    # Les CT sont souvent dans des sous-dossiers (ex: `nifti/`), donc on cherche en récursif.
    ct_files = list(patient_dir.rglob("*ct_post*.nii*"))

    if not ct_files:
        print("  -> CT non trouvé")
        continue

    # Préférer un NIfTI compressé standard si plusieurs candidats existent.
    ct_files_sorted = sorted(
        ct_files,
        key=lambda p: (
            0 if p.name.endswith(".nii.gz") else 1,
            len(p.parts),
            p.name,
        ),
    )
    ct_path = ct_files_sorted[0]
    print("  -> CT trouvé :", ct_path.relative_to(patient_dir))

    contacts = detect_contacts(ct_path)

    if len(contacts) == 0:
        print("  -> Aucun contact détecté")
        continue

    electrodes = cluster_electrodes(contacts)

    output_path = output_dir / f"{patient_dir.name}_predicted_contacts.csv"

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Header
        writer.writerow([
            "electrode_id",
            "contact_id",
            "x",
            "y",
            "z"
        ])

        for e_id, electrode in enumerate(electrodes):

            electrode_name = f"E{e_id+1}"

            for c_id, point in enumerate(electrode):

                writer.writerow([
                    electrode_name,
                    c_id + 1,
                    point[0],
                    point[1],
                    point[2]
                ])

    print("  -> CSV généré :", output_path.name)

print("\nTERMINE.")
