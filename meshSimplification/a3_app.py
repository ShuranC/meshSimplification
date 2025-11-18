from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QApplication
from controls import ControlPanel
from simplification_viewer import SimplificationViewer


class MeshSimplificationApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mesh Simplification - COMP557F25A3 - SHURAN CUI - 261275097")

        main_layout = QHBoxLayout()

        self.view = SimplificationViewer()
        main_layout.addWidget(self.view, stretch=1)
        controls = ControlPanel(self.view)
        self.view.set_update_UI_callback(controls.update_LOD_slider)
        self.view.set_keyboard_callback(controls.keyPressEvent)

        control_panel = QWidget()
        control_layout = QVBoxLayout()
        controls.get_controls(control_layout)
        control_layout.addStretch()  # Push controls to top
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(550)
        main_layout.addWidget(control_panel)

        self.setLayout(main_layout)

        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.timer_update)
        self.anim_timer.start(16)

    def keyPressEvent(self, event):
        self.view.keyPressEvent(event)

    def timer_update(self):
        for child in self.findChildren(QWidget):
            child.update()


app = QApplication([])
window = MeshSimplificationApp()
window.resize(1800, 1000)
window.show()
app.exec_()
