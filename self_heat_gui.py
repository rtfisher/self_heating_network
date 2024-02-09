import sys
import math
from pynucastro.nucdata import Nucleus
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QMessageBox

class IsotopeSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_isotopes = []  # Initialize the list to hold selected isotopes
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Isotope Selector')
        self.grid_layout = QGridLayout()
        
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
