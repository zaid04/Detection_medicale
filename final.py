import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET
from pathlib import Path


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
    ct = nib.load(str(ct_path))
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

    if len(filtered_centers) == 0:
        return {
            "num_centers": 0,
            "num_electrodes": 0,
        }

    # DBSCAN
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(filtered_centers)

    unique_labels = set(labels)
    num_electrodes = len([l for l in unique_labels if l != -1])

    return {
        "num_centers": len(filtered_centers),
        "num_electrodes": num_electrodes,
    }


# =========================
# 3. Paramètres optimisés
# =========================
threshold = 2000
min_size = 12
max_size = 150
dbscan_eps = 10
dbscan_min_samples = 2

x_min, x_max = 80, 430
y_min, y_max = 80, 430
z_min, z_max = 20, 250


# =========================
# 4. Dossier de travail
# =========================
base_dir = Path("/Users/chiki/Downloads/DataHandsOn2")

# Ici, on cherche directement les CT à la racine
ct_files = sorted(base_dir.glob("*ct_post.nii.gz"))

if not ct_files:
    raise FileNotFoundError(f"Aucun fichier *ct_post.nii.gz trouvé dans {base_dir}")

results = []

print("=" * 80)
print("Lancement multi-patients (mode fichiers à plat)")
print("=" * 80)
print(
    f"Paramètres utilisés : threshold={threshold}, "
    f"min_size={min_size}, max_size={max_size}, "
    f"eps={dbscan_eps}, min_samples={dbscan_min_samples}"
)

for ct_path in ct_files:
    patient_id = ct_path.name.replace("_ct_post.nii.gz", "")

    # On teste plusieurs noms XML possibles
    xml_candidates = [
        base_dir / f"{patient_id}_electrodesWithModels.xml",
        base_dir / f"{patient_id}_electrodes_withModels.xml",
        base_dir / f"{patient_id}_electrodeswithModels.xml",
    ]

    xml_path = None
    for cand in xml_candidates:
        if cand.exists():
            xml_path = cand
            break

    print(f"\n--- Patient : {patient_id} ---")
    print(f"  CT  : {ct_path.name}")

    if xml_path is None:
        print("  XML ground truth introuvable -> patient ignoré")
        continue

    print(f"  XML : {xml_path.name}")

    gt_num_electrodes, gt_num_contacts = read_ground_truth(xml_path)

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

    algo_num_centers = out["num_centers"]
    algo_num_electrodes = out["num_electrodes"]

    diff_electrodes = algo_num_electrodes - gt_num_electrodes
    diff_contacts = algo_num_centers - gt_num_contacts

    results.append({
        "patient_id": patient_id,
        "gt_electrodes": gt_num_electrodes,
        "algo_electrodes": algo_num_electrodes,
        "diff_electrodes": diff_electrodes,
        "gt_contacts": gt_num_contacts,
        "algo_contacts": algo_num_centers,
        "diff_contacts": diff_contacts,
    })

    print(f"  Ground truth électrodes : {gt_num_electrodes}")
    print(f"  Algo électrodes         : {algo_num_electrodes}")
    print(f"  Diff électrodes         : {diff_electrodes:+d}")

    print(f"  Ground truth contacts   : {gt_num_contacts}")
    print(f"  Algo contacts           : {algo_num_centers}")
    print(f"  Diff contacts           : {diff_contacts:+d}")


# =========================
# 5. Résumé final
# =========================
print("\n" + "=" * 80)
print("RÉSUMÉ FINAL")
print("=" * 80)

if results:
    header = (
        f"{'Patient':<20}"
        f"{'GT Elec':>10}"
        f"{'Algo Elec':>12}"
        f"{'Diff':>8}"
        f"{'GT Ctc':>10}"
        f"{'Algo Ctc':>12}"
        f"{'Diff':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['patient_id']:<20}"
            f"{r['gt_electrodes']:>10}"
            f"{r['algo_electrodes']:>12}"
            f"{r['diff_electrodes']:>8}"
            f"{r['gt_contacts']:>10}"
            f"{r['algo_contacts']:>12}"
            f"{r['diff_contacts']:>8}"
        )

    mean_abs_elec = np.mean([abs(r["diff_electrodes"]) for r in results])
    mean_abs_ctc = np.mean([abs(r["diff_contacts"]) for r in results])

    print("\nMoyenne erreur absolue électrodes :", round(mean_abs_elec, 2))
    print("Moyenne erreur absolue contacts   :", round(mean_abs_ctc, 2))
else:
    print("Aucun patient traité.")