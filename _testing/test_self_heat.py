import pytest
import numpy as np
import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aux

# Test functions for PYNUCDET code (self-heat)


@pytest.mark.unit
@pytest.mark.helmholtz
class TestHelmholtzEOS:
    """Test suite for Helmholtz EOS calls"""

    def test_helmholtz_inversion_accuracy(self, helmholtz_executable):
        """
        Test that the inverted Helmholtz call on a Helmholtz call returns
        the original values for density and temperature
        """
        # Call Helmholtz with density, temperature, abar, zbar
        rho = 1.0e6
        T = 1.0e9
        abar = 16.0
        zbar = 8.0
        pres = 0.0  # pressure not needed for forward call
        invert = False  # Forward EOS Call

        dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(
            invert, rho, T, abar, zbar, pres
        )

        # Then call Helmholtz with the same density and pressure returned from the first call
        invert = True  # Inverted EOS Call
        dens_inv, pres_inv, eint_inv, gammac_inv, gammae_inv, h_inv, cs_inv, cp_inv, cv_inv = \
            aux.call_helmholtz(invert, dens, T, abar, zbar, pres)

        # Check that density is recovered accurately
        np.testing.assert_allclose(
            dens_inv, rho, rtol=1e-5, atol=1e-8,
            err_msg="Inverted Helmholtz call should recover original density"
        )

    def test_helmholtz_returns_positive_values(self, helmholtz_executable):
        """Test that Helmholtz returns physically valid positive values"""
        rho = 1.0e6
        T = 1.0e9
        abar = 16.0
        zbar = 8.0
        pres = 0.0
        invert = False

        dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(
            invert, rho, T, abar, zbar, pres
        )

        # All physical quantities should be positive
        assert dens > 0, "Density should be positive"
        assert pres > 0, "Pressure should be positive"
        assert eint > 0, "Internal energy should be positive"
        assert cs > 0, "Sound speed should be positive"
        assert cv > 0, "Specific heat cv should be positive"
        assert cp > 0, "Specific heat cp should be positive"
        assert cp >= cv, "cp should be greater than or equal to cv"

    def test_helmholtz_with_test_data(self, helmholtz_executable, test_data_files):
        """
        Test that Helmholtz returns expected values for a set of test data
        """
        input_file_path, output_file_path = test_data_files

        with open(input_file_path, 'r') as file1, open(output_file_path, 'r') as file2:
            # Iterate over each line in the files
            for line_num, (line1, line2) in enumerate(zip(file1, file2), start=1):
                # Strip newline characters and split the lines by spaces to parse columns
                columns1 = line1.strip().split()
                columns2 = line2.strip().split()

                # Convert each string in the columns list to a float and thence to a numpy array
                expected_outputs = np.array([float(column2) for column2 in columns2])

                # Directly convert the first column to a Boolean for file1
                invert = columns1[0] == "True"  # Convert first column directly to Boolean
                rho, T, abar, zbar, pres = [float(column1) for column1 in columns1[1:]]

                # Call Helmholtz and return the results for testing
                dens, pres, eint, gammac, gammae, h, cs, cp, cv = aux.call_helmholtz(
                    invert, rho, T, abar, zbar, pres
                )
                computed_outputs = np.array([dens, pres, eint, gammac, gammae, h, cs, cp, cv])

                np.testing.assert_allclose(
                    computed_outputs, expected_outputs, rtol=1e-5, atol=1e-8,
                    err_msg=f"Test data mismatch at line {line_num}: "
                            f"invert={invert}, rho={rho}, T={T}, abar={abar}, zbar={zbar}, pres={pres}"
                )


@pytest.mark.unit
class TestEOSProperties:
    """Test thermodynamic properties and relationships"""

    @pytest.mark.helmholtz
    def test_pressure_increases_with_temperature(self, helmholtz_executable):
        """Test that pressure increases with temperature at constant density"""
        rho = 1.0e6
        abar = 16.0
        zbar = 8.0
        pres = 0.0
        invert = False

        T_low = 5.0e8
        T_high = 2.0e9

        _, pres_low, _, _, _, _, _, _, _ = aux.call_helmholtz(invert, rho, T_low, abar, zbar, pres)
        _, pres_high, _, _, _, _, _, _, _ = aux.call_helmholtz(invert, rho, T_high, abar, zbar, pres)

        assert pres_high > pres_low, "Pressure should increase with temperature"

    @pytest.mark.helmholtz
    def test_pressure_increases_with_density(self, helmholtz_executable):
        """Test that pressure increases with density at constant temperature"""
        T = 1.0e9
        abar = 16.0
        zbar = 8.0
        pres = 0.0
        invert = False

        rho_low = 1.0e5
        rho_high = 1.0e7

        _, pres_low, _, _, _, _, _, _, _ = aux.call_helmholtz(invert, rho_low, T, abar, zbar, pres)
        _, pres_high, _, _, _, _, _, _, _ = aux.call_helmholtz(invert, rho_high, T, abar, zbar, pres)

        assert pres_high > pres_low, "Pressure should increase with density"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
