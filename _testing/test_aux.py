import pytest
import numpy as np
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aux
import subprocess
from unittest.mock import patch, MagicMock

# Test functions for aux.py module

@pytest.mark.unit
@pytest.mark.helmholtz
class TestHelmholtzCall:
    """Test suite for call_helmholtz function"""

    def test_helmholtz_forward_call(self, helmholtz_executable):
        """Test forward Helmholtz EOS call (isochoric)"""
        rho = 1.0e6
        T = 1.0e9
        abar = 16.0
        zbar = 8.0
        pres = 0.0
        invert = False

        dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(
            invert, rho, T, abar, zbar, pres
        )

        # Check that all outputs are positive
        assert dens > 0, "Density should be positive"
        assert pres > 0, "Pressure should be positive"
        assert eint > 0, "Internal energy should be positive"
        assert cs > 0, "Sound speed should be positive"
        assert cv > 0, "Specific heat cv should be positive"
        assert cp > 0, "Specific heat cp should be positive"

        # Check that density is preserved for isochoric call
        np.testing.assert_allclose(dens, rho, rtol=1e-5)

    def test_helmholtz_inverse_call(self, helmholtz_executable):
        """Test inverse Helmholtz EOS call (isobaric)"""
        # First get pressure from forward call
        rho = 1.0e6
        T = 1.0e9
        abar = 16.0
        zbar = 8.0
        pres_init = 0.0
        invert = False

        dens_fwd, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(
            invert, rho, T, abar, zbar, pres_init
        )

        # Now call inverse
        invert = True
        dens_inv, pres_inv, eint_inv, gammac_inv, gammae_inv, h_inv, cs_inv, cp_inv, cv_inv = \
            aux.call_helmholtz(invert, dens_fwd, T, abar, zbar, pres)

        # Check that pressure is preserved for isobaric call
        np.testing.assert_allclose(pres_inv, pres, rtol=1e-5)

        # Check that density is recovered
        np.testing.assert_allclose(dens_inv, rho, rtol=1e-4)

    def test_helmholtz_different_compositions(self, helmholtz_executable):
        """Test Helmholtz with different compositions"""
        rho = 1.0e5
        T = 1.0e9
        pres = 0.0
        invert = False

        # Test pure hydrogen
        abar_h = 1.0
        zbar_h = 1.0
        dens_h, pres_h, eint_h, gammac_h, gammae_h, h_h, cs_h, cp_h, cv_h = \
            aux.call_helmholtz(invert, rho, T, abar_h, zbar_h, pres)

        # Test pure helium
        abar_he = 4.0
        zbar_he = 2.0
        dens_he, pres_he, eint_he, gammac_he, gammae_he, h_he, cs_he, cp_he, cv_he = \
            aux.call_helmholtz(invert, rho, T, abar_he, zbar_he, pres)

        # Helium should have different thermodynamic properties
        assert pres_h != pres_he, "Different compositions should yield different pressures"

    def test_helmholtz_temperature_variation(self, helmholtz_executable):
        """Test Helmholtz response to temperature changes"""
        rho = 1.0e6
        abar = 16.0
        zbar = 8.0
        pres = 0.0
        invert = False

        T1 = 1.0e8
        T2 = 1.0e9

        dens1, pres1, eint1, gammac1, gammae1, h1, cs1, cp1, cv1 = \
            aux.call_helmholtz(invert, rho, T1, abar, zbar, pres)

        dens2, pres2, eint2, gammac2, gammae2, h2, cs2, cp2, cv2 = \
            aux.call_helmholtz(invert, rho, T2, abar, zbar, pres)

        # Higher temperature should yield higher pressure and energy
        assert pres2 > pres1, "Higher temperature should yield higher pressure"
        assert eint2 > eint1, "Higher temperature should yield higher internal energy"

    def test_helmholtz_executable_not_found(self):
        """Test error handling when Helmholtz executable is missing"""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("Helmholtz executable not found")

            with pytest.raises(FileNotFoundError):
                aux.call_helmholtz(False, 1.0e6, 1.0e9, 16.0, 8.0, 0.0)


@pytest.mark.unit
class TestFloatToLatexScientific:
    """Test suite for float_to_latex_scientific function"""

    def test_zero_value(self):
        """Test conversion of zero"""
        result = aux.float_to_latex_scientific(0)
        assert result == "0", f"Expected '0', got '{result}'"

    def test_scientific_notation_positive(self):
        """Test conversion to scientific notation for large positive numbers"""
        result = aux.float_to_latex_scientific(1.5e20, precision=1)
        assert "1.5" in result, f"Expected '1.5' in result, got '{result}'"
        assert "10^{20}" in result, f"Expected '10^{{20}}' in result, got '{result}'"

    def test_scientific_notation_negative(self):
        """Test conversion to scientific notation for small positive numbers"""
        result = aux.float_to_latex_scientific(3.2e-5, precision=1)
        assert "3.2" in result, f"Expected '3.2' in result, got '{result}'"
        assert "10^{-5}" in result, f"Expected '10^{{-5}}' in result, got '{result}'"

    def test_plain_number_small(self):
        """Test plain number representation for values in [0.01, 99.99]"""
        result = aux.float_to_latex_scientific(5.5, precision=1)
        assert result == "5.5", f"Expected '5.5', got '{result}'"

    def test_plain_number_at_boundaries(self):
        """Test plain number representation at boundary values"""
        # Test lower boundary
        result_low = aux.float_to_latex_scientific(0.01, precision=2)
        assert result_low == "0.01", f"Expected '0.01', got '{result_low}'"

        # Test upper boundary
        result_high = aux.float_to_latex_scientific(99.99, precision=2)
        assert result_high == "99.99", f"Expected '99.99', got '{result_high}'"

    def test_precision_control(self):
        """Test precision parameter control"""
        value = 1234.5

        result_1 = aux.float_to_latex_scientific(value, precision=1)
        assert "1234.5" in result_1

        result_2 = aux.float_to_latex_scientific(value, precision=2)
        assert "1234.50" in result_2

    def test_negative_numbers(self):
        """Test conversion of negative numbers"""
        result = aux.float_to_latex_scientific(-5.5e10, precision=1)
        assert "-5.5" in result
        assert "10^{10}" in result

    def test_very_small_number(self):
        """Test conversion of very small numbers"""
        result = aux.float_to_latex_scientific(1.0e-100, precision=1)
        assert "1.0" in result
        assert "10^{-100}" in result

    def test_very_large_number(self):
        """Test conversion of very large numbers"""
        result = aux.float_to_latex_scientific(9.99e99, precision=2)
        assert "9.99" in result
        assert "10^{99}" in result


@pytest.mark.unit
@pytest.mark.helmholtz
class TestHelmholtzWithTestData:
    """Test Helmholtz against known test data files"""

    def test_helmholtz_test_data(self, helmholtz_executable, test_data_files):
        """Test that Helmholtz returns expected values from test data files"""
        input_file_path = 'helm_input_test_data.txt'
        output_file_path = 'helm_output_test_data.txt'

        with open(input_file_path, 'r') as file1, open(output_file_path, 'r') as file2:
            for line1, line2 in zip(file1, file2):
                columns1 = line1.strip().split()
                columns2 = line2.strip().split()

                expected_outputs = np.array([float(column2) for column2 in columns2])

                invert = columns1[0] == "True"
                rho, T, abar, zbar, pres = [float(column1) for column1 in columns1[1:]]

                dens, pres, eint, gammac, gammae, h, cs, cp, cv = \
                    aux.call_helmholtz(invert, rho, T, abar, zbar, pres)
                computed_outputs = np.array([dens, pres, eint, gammac, gammae, h, cs, cp, cv])

                np.testing.assert_allclose(
                    computed_outputs, expected_outputs,
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"Mismatch for input: invert={invert}, rho={rho}, T={T}, abar={abar}, zbar={zbar}, pres={pres}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
