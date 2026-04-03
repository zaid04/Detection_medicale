"""
EpilocView.py
Module 3D Slicer pour la visualisation et la localisation automatique
des électrodes SEEG.
Version intégrée avec ml.py
"""

import sys
import numpy as np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)

# ─────────────────────────────────────────────────────────────
# Couleurs
# ─────────────────────────────────────────────────────────────
ELECTRODE_COLORS = [
    (1.0, 0.2, 0.2),
    (0.2, 0.6, 1.0),
    (0.2, 0.9, 0.2),
    (1.0, 0.6, 0.0),
    (0.8, 0.2, 0.8),
    (0.0, 0.8, 0.8),
]


# ═══════════════════════════════════════════════════════════════
# MODULE
# ═══════════════════════════════════════════════════════════════

class EpilocView(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "EpiLoc View"
        self.parent.categories = ["SEEG"]
        self.parent.contributors = ["IDMC 2026"]
        self.parent.helpText = "Localisation automatique des électrodes SEEG."


# ═══════════════════════════════════════════════════════════════
# WIDGET
# ═══════════════════════════════════════════════════════════════

class EpilocViewWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = EpilocViewLogic()

        self.patientDirButton = ctk.ctkPathLineEdit()
        self.patientDirButton.filters = ctk.ctkPathLineEdit.Dirs
        self.layout.addWidget(self.patientDirButton)

        self.runButton = qt.QPushButton("Lancer la localisation")
        self.layout.addWidget(self.runButton)

        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setMaximumHeight(150)
        self.layout.addWidget(self.logText)

        self.runButton.connect("clicked(bool)", self.onRunClicked)
        self.layout.addStretch(1)

    def onRunClicked(self):

        patient_dir = self.patientDirButton.currentPath

        if not patient_dir:
            slicer.util.errorDisplay("Veuillez sélectionner un dossier patient.")
            return

        self.logText.clear()
        self._log("Lancement du pipeline...")

        results = self.logic.run_pipeline(patient_dir)

        if results:
            self.logic.display_electrodes(results)
            self._log("Pipeline terminé.")
        else:
            self._log("Aucun résultat produit.")

    def _log(self, message):
        self.logText.appendPlainText(message)
        slicer.app.processEvents()


# ═══════════════════════════════════════════════════════════════
# LOGIC
# ═══════════════════════════════════════════════════════════════

class EpilocViewLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        super().__init__()
        self._markup_nodes = []

    # ─────────────────────────────────────────────────────────
    # PIPELINE ML
    # ─────────────────────────────────────────────────────────
    def run_pipeline(self, patient_dir):

        import importlib

        code_path = "/Users/chiki/Desktop/code"

        if code_path not in sys.path:
            sys.path.append(code_path)

        try:
            import ml
            importlib.reload(ml)
        except Exception as e:
            print("Erreur import ml.py:", e)
            return None

        print("ML chargé.")

        try:
            # 1️⃣ Entraîner le modèle
            model = ml.train_model()
            print("Modèle entraîné.")

            # 2️⃣ Prédire pour CE patient
            electrodes, _ = ml.predict_patient(patient_dir, model)

            print(f"{len(electrodes)} électrodes détectées.")

            return electrodes

        except Exception as e:
            print("Erreur pipeline:", e)
            return None

    # ─────────────────────────────────────────────────────────
    # AFFICHAGE
    # ─────────────────────────────────────────────────────────
    def display_electrodes(self, electrodes):

        self.clear_markups()

        for idx, (elec_id, elec_data) in enumerate(electrodes.items()):

            contacts = elec_data["contacts"]
            color = ELECTRODE_COLORS[idx % len(ELECTRODE_COLORS)]

            markup_node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode"
            )
            markup_node.SetName(elec_id)

            display_node = markup_node.GetDisplayNode()
            if display_node:
                display_node.SetSelectedColor(*color)
                display_node.SetColor(*color)
                display_node.SetGlyphScale(3.0)

            for i, coord in enumerate(contacts):

                # Conversion LPS → RAS
                x = -coord[0]
                y = -coord[1]
                z = coord[2]

                pt_idx = markup_node.AddControlPoint(x, y, z)
                markup_node.SetNthControlPointLabel(
                    pt_idx, f"{elec_id}_{i+1}"
                )

            self._markup_nodes.append(markup_node)

        slicer.util.resetThreeDViews()

    def clear_markups(self):
        nodes = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        for node in nodes:
            slicer.mrmlScene.RemoveNode(node)
        self._markup_nodes = []
        

# ═══════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════

class EpilocViewTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        print("Test OK.")