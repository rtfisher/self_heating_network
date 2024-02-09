import sys
import math
from pynucastro.nucdata import Nucleus
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QMessageBox, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class AxisWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)  # Set minimum size for the drawing area

    def paintEvent(self, event):
        qp = QPainter(self)
#        qp.begin(self)
        self.drawCoordinates(qp)
        qp.end()

    def drawCoordinates(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        # Draw Z axis (Vertical)
        qp.drawLine(20, 80, 20, 20)  # Adjust these coordinates as needed

        # Draw N axis (Horizontal)
        qp.drawLine(20, 80, 80, 80)  # Adjust these coordinates as needed

        # Add labels for Z and N
        qp.drawText(5, 20, 'Z')
        qp.drawText(80, 95, 'N')

class IsotopeSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_isotopes = []  # Initialize the list to hold selected isotopes
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Isotope Selector')
        self.grid_layout = QGridLayout()

        # Instructions included as a Qlabel
        instructions = QLabel("Please deselect any isotopes to exclude from the reaction network by clicking. Green = included, White = excluded. Then press the \"Select Isotopes\" button to continue.")
        instructions.setWordWrap(True)  # Enable word wrap for longer text
        self.grid_layout.addWidget(instructions, 0, 0, 1, -1)  # Span the label across the top

       # Create and add the AxisWidget to the layout
        self.axisWidget = AxisWidget()
        self.grid_layout.addWidget(self.axisWidget, 1, 0)  # Add it to the upper left
        
        # Your provided list of isotopes
        self.isotope_list = [
            "p", "n", "he4", "b11", "c12", "c13", "n13", "n14",
            "n15", "o15", "o16", "o17", "f18", "ne19",
            "ne20", "ne21", "ne22", "na22", "na23", "mg23", "mg24",
            "mg25", "mg26", "al25", "si26", "al26", "al27", "si28", "si29",
            "si30", "p29", "p30", "s30", "p31", "s31",
            "s32", "s33", "ar34", "cl33", "cl34", "cl35", "ar36", "ar37",
            "ar38", "ar39", "k39", "ca40", "sc43", "ti44", "v47",
            "cr48", "mn51", "fe52", "fe55", "co55", "ni56",
            "ni58", "ni59" 
        ]

        # By default, all isotopes are selected
        self.selected_isotopes = self.isotope_list.copy()

        # Initialize the mapping for isotopes to Z and N values
        zn_map = {iso: Nucleus(iso) for iso in self.isotope_list}
        
        # Calculate the maximum Z to invert the grid placement
        max_Z = max(nuc.Z for nuc in zn_map.values())

        # Sort isotopes based on Z (vertical) and N (horizontal)
#        sorted_isotopes = sorted(self.isotope_list, key=lambda x: (zn_map[x][0], zn_map[x][1]))

# Place buttons in the grid based on Z and N, inverting Z
        for iso, nuc in zn_map.items():
            Z, N = nuc.Z, nuc.N  # Correctly access Z and N
            btn = QPushButton(iso)
            btn.setCheckable(True)
            btn.setChecked(True)  # All isotopes selected by default
            btn.setStyleSheet("background-color: lightgreen;")
            btn.clicked.connect(self.toggleIsotope)
            self.grid_layout.addWidget(btn, max_Z - Z, N)  # Invert Z placement by subtracting from max_Z

        # Button to finalize selection and close the GUI
        self.include_button = QPushButton('Include Isotopes')
        self.include_button.clicked.connect(self.finalizeSelection)
#        self.grid_layout.addWidget(self.include_button, grid_size, 0, 1, grid_size) # Span the button across the last row
#        self.grid_layout.addWidget(self.include_button, max(zn_map.values(), key=lambda x: x[0])[0] + 1, 0, 1, max_N + 1)
        self.grid_layout.addWidget(self.include_button, max_Z + 1, 0)


        
        self.setLayout(self.grid_layout)
    
    def toggleIsotope(self):
        button = self.sender()
        isotope = button.text()
        
        if button.isChecked():
            button.setStyleSheet("background-color: lightgreen;")
            if isotope not in self.selected_isotopes:
                self.selected_isotopes.append(isotope)
        else:
            button.setStyleSheet("")
            if isotope in self.selected_isotopes:
                self.selected_isotopes.remove(isotope)
    
    def finalizeSelection(self):
        QMessageBox.information(self, "Selected Isotopes", ", ".join(self.selected_isotopes))
        self.close()  # Close the GUI after finalizing the selection
