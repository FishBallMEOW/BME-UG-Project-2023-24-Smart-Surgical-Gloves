import sys
from PyQt6.QtWidgets import QApplication,  QMessageBox, QWidget, QHBoxLayout,  QPushButton
import pyglet 
from pyglet.gl import *

class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_bool = False
        self.setWindowTitle('Control')
        self.setGeometry(100, 100, 300, 100)

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
            'Information',
            'This is important information.'
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