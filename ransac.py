import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.decomposition import PCA

# =========================
# 1. Charger le CT
# =========================
path = "/Users/chiki/Downloads/DataHandsOn2/pat_03601_2081_ct_post.nii.gz"

ct = nib.load(path)
data = ct.get_fdata()

print("Shape:", data.shape)
print("Min:", np.min(data))
print("Max:", np.max(data))

# =========================
# 2. Paramètres
# =========================
slices_to_check = [40, 80, 120, 160, 200, 240]

threshold = 2000
min_size = 12
max_size = 150

x_min, x_max = 80, 430
y_min, y_max = 80, 430
z_min, z_max = 20, 250

# Paramètres RANSAC
distance_threshold = 4
min_inliers = 5
max_iterations = 500

# =========================
# 3. Détection des centres candidats
# =========================
binary = data > threshold
labeled, num_features = ndimage.label(binary)

print("Nombre total de composantes 3D :", num_features)

component_sizes = np.bincount(labeled.ravel())

valid_labels = []
for label_id, size in enumerate(component_sizes):
    if label_id == 0:
        continue
    if min_size <= size <= max_size:
        valid_labels.append(label_id)

print("Nombre de composantes gardées :", len(valid_labels))

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

print("Nombre de centres après filtrage :", len(filtered_centers))

# =========================
# 4. Fonctions RANSAC 3D
# =========================
def point_line_distance(points, p1, p2):
    """
    Distance de chaque point à la droite définie par p1 et p2.
    points: array (N,3)
    p1, p2: arrays (3,)
    """
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


def fit_ransac_lines(points, distance_threshold=6, min_inliers=4, max_iterations=500):
    """
    Détecte plusieurs lignes 3D par RANSAC.
    Retourne une liste de clusters, chaque cluster = points d'une électrode.
    """
    remaining_points = points.copy()
    electrodes = []

    while len(remaining_points) >= min_inliers:
        best_inliers = None
        best_count = 0

        for _ in range(max_iterations):
            if len(remaining_points) < 2:
                break

            idx = np.random.choice(len(remaining_points), size=2, replace=False)
            p1, p2 = remaining_points[idx[0]], remaining_points[idx[1]]

            distances = point_line_distance(remaining_points, p1, p2)
            inliers_mask = distances < distance_threshold
            inliers = remaining_points[inliers_mask]

            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers_mask

        if best_inliers is None or best_count < min_inliers:
            break

        cluster_points = remaining_points[best_inliers]
        electrodes.append(cluster_points)

        # retirer les points utilisés
        remaining_points = remaining_points[~best_inliers]

    return electrodes, remaining_points


# =========================
# 5. RANSAC sur les centres
# =========================
electrodes, remaining_noise = fit_ransac_lines(
    filtered_centers,
    distance_threshold=distance_threshold,
    min_inliers=min_inliers,
    max_iterations=max_iterations,
)

print("Nombre d'électrodes détectées par RANSAC :", len(electrodes))
print("Nombre de points restants (bruit) :", len(remaining_noise))

for i, cluster in enumerate(electrodes, start=1):
    print(f"\nÉlectrode {i}: {len(cluster)} contacts")
    for j, p in enumerate(cluster, start=1):
        print(f"  Contact {j}: {p}")

# =========================
# 6. Ordonner les contacts avec PCA
# =========================
ordered_electrodes = {}

print("\n=== Ordre des contacts par électrode (PCA) ===")

for i, cluster_points in enumerate(electrodes, start=1):
    if len(cluster_points) < 2:
        continue

    pca = PCA(n_components=1)
    projected = pca.fit_transform(cluster_points).ravel()
    order = np.argsort(projected)
    ordered_points = cluster_points[order]

    ordered_electrodes[i] = ordered_points

    print(f"\nÉlectrode {i} ordonnée :")
    for j, p in enumerate(ordered_points, start=1):
        print(f"  Contact {j}: {p}")

# =========================
# 7. Affichage des centres sur plusieurs slices
# =========================
colors = ["r", "g", "b", "c", "m", "y", "w"]

for idx in slices_to_check:
    plt.figure(figsize=(6, 6))
    plt.imshow(data[:, :, idx], cmap="gray")

    for i, points in ordered_electrodes.items():
        color = colors[(i - 1) % len(colors)]
        for j, p in enumerate(points, start=1):
            x, y, z = p
            if abs(z - idx) < 1.5:
                plt.plot(y, x, marker="o", color=color, markersize=5)
                plt.text(y + 2, x + 2, str(j), color="yellow", fontsize=8)

    plt.title(f"Slice {idx} avec électrodes RANSAC")
    plt.axis("off")
    plt.show()