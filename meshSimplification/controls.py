from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QLabel, QPushButton, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from simplification_viewer import SimplificationViewer
from pathlib import Path


class ControlPanel:
    def __init__(self, viewer: SimplificationViewer):
        self.viewer = viewer

        # The options available in the control panel:
        self.load_button = None  # Load a new mesh
        self.twin_button = None  # Move to the twin of the current HE
        self.next_button = None  # Move to the next HE
        self.collapse_button = None  # Collapse the current HE
        self.jump_button = None  # Jump to the best HE to collapse
        self.best_button = None  # Collapse the best HE
        self.collapse_all_button = None  # Collapse the model until there are no more edges left
        self.LOD_slider = None  # Select the level of detail (# edges collapsed)
        # Number of faces / vertices at the current LOD
        self.faces_label = None
        self.verts_label = None
        # Visualization options
        self.draw_vertex_IDs = None
        self.draw_face_IDs = None
        self.draw_current_he = None
        self.wireframe = None

    def get_controls(self, layout):
        layout.addWidget(QLabel("Controls:"))
        layout.addWidget(QLabel("  Mouse drag: rotate view"))
        layout.addWidget(QLabel("  Mouse wheel: zoom in/out"))

        self.load_button = Button(layout, "Load Mesh", lambda: self.on_load_pressed())
        self.next_button = Button(layout, "Next half-edge (N)", lambda: self.viewer.next_half_edge())
        self.twin_button = Button(layout, "Twin half-edge (T)", lambda: self.viewer.twin_half_edge())
        self.collapse_button = Button(layout, "Collapse current half-edge (C)",
                                      lambda: self.viewer.collapse_current_half_edge())
        self.jump_button = Button(layout, "Jump to best half-edge (J)", lambda: self.viewer.jump_to_best_edge())
        self.best_button = Button(layout, "Collapse best half-edge (B)", lambda: self.viewer.collapse_best_edge())
        self.collapse_all_button = Button(layout, "Collapse all (A)", lambda: self.viewer.collapse_all_in_order())
        self.draw_face_IDs = CheckboxControl(layout, "draw face IDs (F)", self.viewer.draw_face_IDs,
                                             lambda x: setattr(self.viewer, 'draw_face_IDs', x == Qt.Checked))
        self.draw_vertex_IDs = CheckboxControl(layout, "draw vertex IDs (V)", self.viewer.draw_vertex_IDs,
                                               lambda x: setattr(self.viewer, 'draw_vertex_IDs', x == Qt.Checked))
        self.wireframe = CheckboxControl(layout, "mesh wireframe (W)", self.viewer.mesh_wireframe,
                                         lambda x: setattr(self.viewer, 'mesh_wireframe', x == Qt.Checked))
        self.draw_current_he = CheckboxControl(layout, "draw current half-edge (H)", self.viewer.draw_current_he,
                                               lambda x: setattr(self.viewer, 'draw_current_he', x == Qt.Checked))
        self.scale_with_LOD = CheckboxControl(layout, "scale mesh with LOD (S)", self.viewer.scale_with_LOD,
                                               lambda x: setattr(self.viewer, 'scale_with_LOD', x == Qt.Checked))
        self.LOD_slider = SliderControl(layout, "level of detail", 0, 0, 0,
                                        lambda x: self.on_LOD_changed(x), scale=1, digits=0)

        self.verts_label = QLabel("Vertices: 0")
        layout.addWidget(self.verts_label)
        self.faces_label = QLabel("Faces: 0")
        layout.addWidget(self.faces_label)

    def on_load_pressed(self):
        current_dir = Path(__file__).parent / "data"  # glsl folder in same directory as this code
        filename, _ = QFileDialog.getOpenFileName(
            None,  # parent widget
            "Open File",  # dialog title
            str(current_dir),  # starting directory
            "obj files (*.obj)"  # file filters
        )
        if filename:
            self.viewer.load_mesh_from_file(filename)

    def on_LOD_changed(self, value):
        self.viewer.set_LOD(value)
        self.verts_label.setText(f"Vertices: {self.viewer.get_vertex_count()}")
        self.faces_label.setText(f"Faces: {self.viewer.get_face_count()}")

    def update_LOD_slider(self):
        self.LOD_slider.setMaxValue(self.viewer.max_LOD)
        self.LOD_slider.setValue(self.viewer.current_LOD)
        self.verts_label.setText(f"Vertices: {self.viewer.get_vertex_count()}")
        self.faces_label.setText(f"Faces: {self.viewer.get_face_count()}")

    def keyPressEvent(self, event: QKeyEvent):
        match event.key():
            case QtCore.Qt.Key.Key_N:
                self.viewer.next_half_edge()
            case QtCore.Qt.Key.Key_T:
                self.viewer.twin_half_edge()
            case QtCore.Qt.Key.Key_C:  # collapse the current half_edge
                self.viewer.collapse_current_half_edge()
            case QtCore.Qt.Key.Key_J:  # go to the best edge
                self.viewer.jump_to_best_edge()
            case QtCore.Qt.Key.Key_B:  # collapse the current half_edge
                self.viewer.collapse_best_edge()
            case QtCore.Qt.Key.Key_A:
                self.viewer.collapse_all_in_order()
            case QtCore.Qt.Key.Key_BracketLeft:
                self.LOD_slider.setValue(max(0, self.viewer.current_LOD - 1))                
            case QtCore.Qt.Key.Key_BracketRight:
                self.LOD_slider.setValue(min(self.viewer.max_LOD, self.viewer.current_LOD + 1))                
            case QtCore.Qt.Key.Key_W:
                self.wireframe.setChecked(not self.wireframe.isChecked())
            case QtCore.Qt.Key.Key_V:
                self.draw_vertex_IDs.setChecked(not self.draw_vertex_IDs.isChecked())
            case QtCore.Qt.Key.Key_F:
                self.draw_face_IDs.setChecked(not self.draw_face_IDs.isChecked())
            case QtCore.Qt.Key.Key_H:
                self.draw_current_he.setChecked(not self.draw_current_he.isChecked())
            case QtCore.Qt.Key.Key_S:
                self.scale_with_LOD.setChecked(not self.scale_with_LOD.isChecked())


class SliderControl(QtWidgets.QWidget):
    """Wrapper for creating sliders in UI."""

    def __init__(self, parent, label, min_val, max_val, init_val, callback1, scale=1.0, digits=2):
        super().__init__()
        self.callback_val_update = callback1
        self.scale = scale
        self.value = init_val
        self.digits = digits
        layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val / scale), int(max_val / scale))
        self.slider.setValue(int(init_val / scale))
        self.slider.valueChanged.connect(self.on_value_changed)
        self.value_label = QtWidgets.QLabel(f"{init_val:.{self.digits}f}")
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        self.setLayout(layout)
        parent.addWidget(self)

    def setMaxValue(self, max_val):
        self.slider.setMaximum(int(max_val / self.scale))

    def getValue(self):
        return self.value

    def setValue(self, val):
        self.slider.setValue(int(val / self.scale))
        self.value_label.setText(f"{val:.{self.digits}f}")

    def on_value_changed(self, value_scaled):
        self.value = value_scaled * self.scale
        self.value_label.setText(f"{self.value:.{self.digits}f}")
        if self.callback_val_update is not None:
            self.callback_val_update(self.value)


class Button(QPushButton):
    """Wrapper for creating buttons in UI."""

    def __init__(self, parent, label, callback=None):
        super().__init__(label)
        if callback is not None:
            self.clicked.connect(callback)
        parent.addWidget(self)


class CheckboxControl(QtWidgets.QWidget):
    """Wrapper for creating labeled check box in UI."""

    def __init__(self, parent, label, init_val: bool, callback=None):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel(label)
        self.box = QtWidgets.QCheckBox()
        self.box.setChecked(init_val)
        if callback is not None:
            self.box.stateChanged.connect(callback)
        layout.addWidget(self.box)
        layout.addWidget(self.label, stretch=1)
        layout.addStretch()  # Push to left
        self.setLayout(layout)
        parent.addWidget(self)

    def isChecked(self):
        return self.box.isChecked()

    def setChecked(self, val: bool):
        self.box.setChecked(val)