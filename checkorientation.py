import nibabel as nib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# =========================================================
# CHEMINS (adaptés à ta structure)
# =========================================================

ct_path = Path("/Users/chiki/Desktop/code/data/pat_03415_1893/nifti/pat_03415_1893_ct_post.nii.gz")

xml_path = Path("/Users/chiki/Desktop/code/data/pat_03415_1893/pat_03415_1893_electrodes_withModels.xml")


# =========================================================
# LECTURE CT
# =========================================================

ct = nib.load(str(ct_path))

print("====================================")
print("Orientation CT :", nib.aff2axcodes(ct.affine))
print("Affine CT :\n", ct.affine)
print("====================================\n")


# =========================================================
# LECTURE XML
# =========================================================

tree = ET.parse(str(xml_path))
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

gt_points = np.array(gt_points)

print("Nombre points XML :", len(gt_points))

if len(gt_points) > 0:
    print("Exemple coord XML :", gt_points[0])
    print("Min XML :", gt_points.min(axis=0))
    print("Max XML :", gt_points.max(axis=0))

print("\n====================================")
print("FIN VERIFICATION")
print("====================================")