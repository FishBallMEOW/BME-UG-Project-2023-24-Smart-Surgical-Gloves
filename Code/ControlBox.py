import sys
from PyQt6.QtWidgets import QApplication,  QMessageBox, QWidget, QHBoxLayout,  QPushButton
import pyglet 
from pyglet.gl import *

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_bool = False
        self.setWindowTitle('Control')
        self.setGeometry(1025, 25, 500, 70)

        layout = QHBoxLayout()
        self.setLayout(layout)

        btn_question = QPushButton('Quit')
        btn_question.clicked.connect(self.question)

        btn_instr = QPushButton('Instruction')
        btn_instr.clicked.connect(self.instruction)

        layout.addWidget(btn_question)
        layout.addWidget(btn_instr)

        self.show()

    def instruction(self):
        QMessageBox.information(
            self,
            'Instruction',
            'Left Panel(3D Graphics Simulation):' + 
            '\nMovement: Forward(W), Backward(S), Right(D), Left(A), Up(Space/UpArrow), Down(Left Shift/DownArrow) \nZoom: Scroll \nRotate: Drag and move ' + 
            '\n\nBottom Right Panel (Stress Strain Plot): \nTool(s): Clear(Clear components in the plot), Fitting Model(Choose the fitting model, Default Linear Model)' + 
            '\nThe calculated stiffness of the current press is print in the title of the plot. As a reference, values of normal and cancerous stiffness[1] are given at the bottom of the panel.' + 
            '\n\n[1]K. Hoyt et al., “Tissue elasticity properties as biomarkers for prostate cancer,” Cancer biomarkers : section A of Disease markers, vol. 4, no. 4–5, pp. 213–25, 2008, doi: https://doi.org/10.3233/cbm-2008-44-505.'
        )

    def question(self):
        answer = QMessageBox.question(
            self,
            'Confirmation',
            'Do you want to quit?',
            QMessageBox.StandardButton.Yes |
            QMessageBox.StandardButton.No
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.close()
            self.close_bool = True
            
    def return_close_bool(self):
        return self.close_bool