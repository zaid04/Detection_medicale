import xml.etree.ElementTree as ET
from pathlib import Path


def read_electrodes_with_models(xml_path: str):
    """
    Lit un fichier electrodesWithModels.xml et retourne :
    - patient_id
    - num_electrodes déclaré dans le XML
    - liste des électrodes avec leurs contacts
    """
    xml_file = Path(xml_path)

    if not xml_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {xml_file}")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    patient_id = root.attrib.get("patientId", "unknown")
    num_electrodes_declared = int(root.attrib.get("numElectrodes", 0))

    electrodes = []

    for electrode in root.findall("Electrode"):
        procedure_name = electrode.attrib.get("procedure", "unknown")
        laterality = electrode.attrib.get("laterality", "unknown")
        is_micro = electrode.attrib.get("isMicro", "unknown")
        electrode_model = electrode.attrib.get("electrodeModel", "unknown")

        target_point = electrode.find("TargetPoint")
        entry_point = electrode.find("EntryPoint")

        target = None
        if target_point is not None:
            target = {
                "x": float(target_point.attrib.get("x", 0)),
                "y": float(target_point.attrib.get("y", 0)),
                "z": float(target_point.attrib.get("z", 0)),
            }

        entry = None
        if entry_point is not None:
            entry = {
                "x": float(entry_point.attrib.get("x", 0)),
                "y": float(entry_point.attrib.get("y", 0)),
                "z": float(entry_point.attrib.get("z", 0)),
            }

        contacts = []
        plots = electrode.find("Plots")
        if plots is not None:
            for plot in plots.findall("Plot"):
                contacts.append(
                    {
                        "number": int(plot.attrib.get("number", 0)),
                        "x": float(plot.attrib.get("x", 0)),
                        "y": float(plot.attrib.get("y", 0)),
                        "z": float(plot.attrib.get("z", 0)),
                    }
                )

        electrodes.append(
            {
                "procedure": procedure_name,
                "laterality": laterality,
                "isMicro": is_micro,
                "electrodeModel": electrode_model,
                "target": target,
                "entry": entry,
                "contacts": contacts,
            }
        )

    return {
        "patient_id": patient_id,
        "num_electrodes_declared": num_electrodes_declared,
        "electrodes": electrodes,
    }


def print_summary(data: dict):
    electrodes = data["electrodes"]
    total_contacts = sum(len(e["contacts"]) for e in electrodes)

    print("=" * 60)
    print(f"Patient ID : {data['patient_id']}")
    print(f"Nombre d'électrodes déclaré dans le XML : {data['num_electrodes_declared']}")
    print(f"Nombre d'électrodes lues : {len(electrodes)}")
    print(f"Nombre total de contacts : {total_contacts}")
    print("=" * 60)

    for i, electrode in enumerate(electrodes, start=1):
        print(f"\nÉlectrode {i}")
        print(f"  Nom / procedure : {electrode['procedure']}")
        print(f"  Latéralité      : {electrode['laterality']}")
        print(f"  isMicro         : {electrode['isMicro']}")
        print(f"  Modèle          : {electrode['electrodeModel']}")

        if electrode["target"] is not None:
            t = electrode["target"]
            print(f"  TargetPoint     : ({t['x']:.2f}, {t['y']:.2f}, {t['z']:.2f})")

        if electrode["entry"] is not None:
            e = electrode["entry"]
            print(f"  EntryPoint      : ({e['x']:.2f}, {e['y']:.2f}, {e['z']:.2f})")

        print(f"  Nombre de contacts : {len(electrode['contacts'])}")

        for contact in electrode["contacts"]:
            print(
                f"    Contact {contact['number']}: "
                f"({contact['x']:.2f}, {contact['y']:.2f}, {contact['z']:.2f})"
            )


if __name__ == "__main__":
    # Remplace ce chemin par TON fichier electrodesWithModels.xml
    xml_path = "/Users/chiki/Downloads/DataHandsOn2/pat_03601_2081_electrodes_withModels.xml"

    data = read_electrodes_with_models(xml_path)
    print_summary(data)