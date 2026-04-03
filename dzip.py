import zipfile
import os
import glob

dossier_source = "/Users/chiki/Desktop/code/dataProjetIDMC"          # Dossier contenant les .zip
dossier_destination = "./extraits"     # Dossier de sortie

os.makedirs(dossier_destination, exist_ok=True)

for fichier in glob.glob(os.path.join(dossier_source, "*.zip")):
    with zipfile.ZipFile(fichier, 'r') as zip_ref:
        zip_ref.extractall(dossier_destination)
        print(f"✅ Extrait : {fichier}")