import sys
import math
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
            "mg25", "mg26", "al25", "al26", "al27", "si28", "si29",
            "si30", "p29", "p30", "p31", "s31",
            "s32", "s33", "cl33", "cl34", "cl35", "ar36", "ar37",
            "ar38", "ar39", "k39", "ca40", "sc43", "ti44", "v47",
            "cr48", "mn51", "fe52", "fe55", "co55", "ni56",
            "ni58", "ni59", "si26", "s30", "ar34"
        ]

        # By default, all isotopes are selected
        self.selected_isotopes = self.isotope_list.copy()
        
        # Determine the grid size
        grid_size = math.ceil(math.sqrt(len(self.isotope_list)))
        
        # Place buttons in a grid, with all buttons selected by default
        for i, isotope in enumerate(self.isotope_list):
            btn = QPushButton(isotope)
            btn.setCheckable(True)
            btn.setChecked(True)  # Button starts in a checked state
            btn.setStyleSheet("background-color: lightgreen;")  # Initially highlighted
            btn.clicked.connect(self.toggleIsotope)
            row = i // grid_size
            col = i % grid_size
            self.grid_layout.addWidget(btn, row, col)
        
        # Button to finalize selection and close the GUI
        self.include_button = QPushButton('Include Isotopes')
        self.include_button.clicked.connect(self.finalizeSelection)
        self.grid_layout.addWidget(self.include_button, grid_size, 0, 1, grid_size) # Span the button across the last row
        
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
