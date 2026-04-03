import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
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
# 2. Détection des centres candidats
# =========================
def detect_candidate_centers(
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
):
    ct = nib.load(str(ct_path))
    data = ct.get_fdata()

    binary = data > threshold
    labeled, num_features = ndimage.label(binary)

    component_sizes = np.bincount(labeled.ravel())

    valid_labels = []
    for label_id, size in enumerate(component_sizes):
        if label_id == 0:
            continue
        if min_size <= size <= max_size:
            valid_labels.append(label_id)

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

    return {
        "data_shape": data.shape,
        "num_features": num_features,
        "num_valid_components": len(valid_labels),
        "filtered_centers": filtered_centers,
    }


# =========================
# 3. Distance point-droite
# =========================
def point_line_distance(points, p1, p2):
    line_vec = p2 - p1
    norm = np.linalg.norm(line_vec)

    if norm < 1e-8:
        return np.full(len(points), np.inf)

    line_unit = line_vec / norm
    vecs = points - p1
    projections = np.outer(np.dot(vecs, line_unit), line_unit)
    orthogonal = vecs - projections
    distances = np.linalg.norm(orthogonal, axis=1)

    return distances


# =========================
# 4. RANSAC multi-lignes
# =========================
def fit_ransac_lines(points, distance_threshold=4, min_inliers=4, max_iterations=500):
    remaining_points = points.copy()
    electrodes = []

    while len(remaining_points) >= min_inliers:
        best_inliers_mask = None
        best_count = 0

        for _ in range(max_iterations):
            if len(remaining_points) < 2:
                break

            idx = np.random.choice(len(remaining_points), size=2, replace=False)
            p1, p2 = remaining_points[idx[0]], remaining_points[idx[1]]

            distances = point_line_distance(remaining_points, p1, p2)
            inliers_mask = distances < distance_threshold
            count = np.sum(inliers_mask)

            if count > best_count:
                best_count = count
                best_inliers_mask = inliers_mask

        if best_inliers_mask is None or best_count < min_inliers:
            break

        cluster_points = remaining_points[best_inliers_mask]
        electrodes.append(cluster_points)

        remaining_points = remaining_points[~best_inliers_mask]

    return electrodes, remaining_points


# =========================
# 5. Ordonner les contacts avec PCA
# =========================
def order_electrodes_with_pca(electrodes):
    ordered_electrodes = []

    for cluster_points in electrodes:
        if len(cluster_points) < 2:
            ordered_electrodes.append(cluster_points)
            continue

        pca = PCA(n_components=1)
        projected = pca.fit_transform(cluster_points).ravel()
        order = np.argsort(projected)
        ordered_points = cluster_points[order]

        ordered_electrodes.append(ordered_points)

    return ordered_electrodes


# =========================
# 6. Traitement d'un patient
# =========================
def process_patient(
    ct_path,
    xml_path,
    threshold,
    min_size,
    max_size,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max,
    distance_threshold,
    min_inliers,
    max_iterations,
):
    gt_num_electrodes, gt_num_contacts = read_ground_truth(xml_path)

    det = detect_candidate_centers(
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
    )

    filtered_centers = det["filtered_centers"]

    electrodes, remaining_noise = fit_ransac_lines(
        filtered_centers,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        max_iterations=max_iterations,
    )

    ordered_electrodes = order_electrodes_with_pca(electrodes)

    algo_num_electrodes = len(ordered_electrodes)
    algo_num_contacts = sum(len(e) for e in ordered_electrodes)

    return {
        "gt_num_electrodes": gt_num_electrodes,
        "gt_num_contacts": gt_num_contacts,
        "num_features": det["num_features"],
        "num_valid_components": det["num_valid_components"],
        "num_candidate_centers": len(filtered_centers),
        "algo_num_electrodes": algo_num_electrodes,
        "algo_num_contacts": algo_num_contacts,
        "remaining_noise": len(remaining_noise),
        "ordered_electrodes": ordered_electrodes,
    }


# =========================
# 7. Paramètres retenus
# =========================
threshold = 2000
min_size = 12
max_size = 150

x_min, x_max = 80, 430
y_min, y_max = 80, 430
z_min, z_max = 20, 250

# Paramètres RANSAC retenus
distance_threshold = 4
min_inliers = 4
max_iterations = 500


# =========================
# 8. Base directory
# =========================
base_dir = Path("/Users/chiki/Downloads/DataHandsOn2")

# Ici on prend les CT à la racine
ct_files = sorted(base_dir.glob("*ct_post.nii.gz"))

if not ct_files:
    raise FileNotFoundError(f"Aucun fichier *ct_post.nii.gz trouvé dans {base_dir}")

results = []

print("=" * 90)
print("RANSAC multi-patients")
print("=" * 90)
print(
    f"Paramètres : threshold={threshold}, min_size={min_size}, max_size={max_size}, "
    f"distance_threshold={distance_threshold}, min_inliers={min_inliers}, "
    f"max_iterations={max_iterations}"
)

for ct_path in ct_files:
    patient_id = ct_path.name.replace("_ct_post.nii.gz", "")

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
    print(f"CT  : {ct_path.name}")

    if xml_path is None:
        print("XML ground truth introuvable -> patient ignoré")
        continue

    print(f"XML : {xml_path.name}")

    out = process_patient(
        ct_path=ct_path,
        xml_path=xml_path,
        threshold=threshold,
        min_size=min_size,
        max_size=max_size,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        distance_threshold=distance_threshold,
        min_inliers=min_inliers,
        max_iterations=max_iterations,
    )

    diff_electrodes = out["algo_num_electrodes"] - out["gt_num_electrodes"]
    diff_contacts = out["algo_num_contacts"] - out["gt_num_contacts"]

    print(f"GT électrodes        : {out['gt_num_electrodes']}")
    print(f"Algo électrodes      : {out['algo_num_electrodes']}")
    print(f"Diff électrodes      : {diff_electrodes:+d}")

    print(f"GT contacts          : {out['gt_num_contacts']}")
    print(f"Algo contacts        : {out['algo_num_contacts']}")
    print(f"Diff contacts        : {diff_contacts:+d}")

    print(f"Centres candidats    : {out['num_candidate_centers']}")
    print(f"Bruit restant        : {out['remaining_noise']}")

    results.append({
        "patient_id": patient_id,
        "gt_electrodes": out["gt_num_electrodes"],
        "algo_electrodes": out["algo_num_electrodes"],
        "diff_electrodes": diff_electrodes,
        "gt_contacts": out["gt_num_contacts"],
        "algo_contacts": out["algo_num_contacts"],
        "diff_contacts": diff_contacts,
        "candidate_centers": out["num_candidate_centers"],
        "remaining_noise": out["remaining_noise"],
    })


# =========================
# 9. Résumé final
# =========================
print("\n" + "=" * 90)
print("RÉSUMÉ FINAL")
print("=" * 90)

if results:
    header = (
        f"{'Patient':<20}"
        f"{'GT Elec':>10}"
        f"{'Algo Elec':>12}"
        f"{'Diff':>8}"
        f"{'GT Ctc':>10}"
        f"{'Algo Ctc':>12}"
        f"{'Diff':>8}"
        f"{'Noise':>10}"
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
            f"{r['remaining_noise']:>10}"
        )

    mean_abs_elec = np.mean([abs(r["diff_electrodes"]) for r in results])
    mean_abs_ctc = np.mean([abs(r["diff_contacts"]) for r in results])

    print("\nMoyenne erreur absolue électrodes :", round(mean_abs_elec, 2))
    print("Moyenne erreur absolue contacts   :", round(mean_abs_ctc, 2))
else:
    print("Aucun patient traité.")