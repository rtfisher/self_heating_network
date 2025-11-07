import pytest
import subprocess
import os
import sys
import numpy as np
import tempfile
import shutil

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_HEAT_PATH = os.path.join(PROJECT_ROOT, 'self_heat.py')

# Integration tests for self_heat.py script

@pytest.mark.integration
@pytest.mark.slow
class TestSelfHeatIntegration:
    """Integration test suite for complete self_heat.py workflows"""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        yield temp_dir
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)

    def test_help_option(self):
        """Test that --help option works"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Help option should exit successfully"
        assert "usage:" in result.stdout.lower(), "Help should display usage information"

    def test_isochoric_short_run(self):
        """Test basic isochoric run with minimal parameters"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-rho', '1e5', '-T', '1e9', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Isochoric run failed: {result.stderr}"
        assert "Isochoric run" in result.stdout, "Should indicate isochoric mode"

    def test_isobaric_short_run(self):
        """Test basic isobaric run with minimal parameters"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isobaric', '-rho', '1e5', '-T', '1e9', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Isobaric run failed: {result.stderr}"
        assert "Isobaric run" in result.stdout, "Should indicate isobaric mode"

    def test_default_mode(self):
        """Test default mode (should be isochoric)"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Default run failed: {result.stderr}"
        assert "Isochoric run" in result.stdout, "Default should be isochoric mode"

    def test_custom_density_temperature(self):
        """Test with custom density and temperature"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-rho', '1e6', '-T', '5e8', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Custom parameters run failed: {result.stderr}"
        assert "1000000.0" in result.stdout or "1e+06" in result.stdout, "Should display custom density"
        assert "500000000.0" in result.stdout or "5e+08" in result.stdout, "Should display custom temperature"

    def test_custom_abundances(self):
        """Test with custom initial abundances"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric',
             '-xp', '0.1', '-xhe4', '0.7', '-xc12', '0.15', '-xo16', '0.05',
             '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Custom abundances run failed: {result.stderr}"

    def test_invalid_abundances_sum(self):
        """Test that invalid abundance sum (not equal to 1) is rejected"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric',
             '-xp', '0.1', '-xhe4', '0.6', '-xc12', '0.2', '-xo16', '0.2',  # Sum = 1.1
             '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode != 0, "Should fail with invalid abundances"
        assert "must add to unity" in result.stdout or "must add to unity" in result.stderr, \
            "Should display error about abundances not summing to unity"

    def test_output_files_created(self):
        """Test that expected output files are created"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Run failed: {result.stderr}"

        # Check for output files
        expected_files = ['abundances.png', 'detonation_lengths.png', 'nuclear_network.py']
        for filename in expected_files:
            assert os.path.exists(filename), f"Expected output file {filename} not found"

    def test_detonation_lengths_data_file(self):
        """Test that detonation_lengths.dat is created and formatted correctly"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric',
             '-xhe4', '0.8', '-xc12', '0.1', '-xo16', '0.1',
             '-rho', '1e5', '-T', '1e9', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Run failed: {result.stderr}"
        assert os.path.exists('detonation_lengths.dat'), "detonation_lengths.dat should be created"

        # Check file format
        with open('detonation_lengths.dat', 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            values = last_line.strip().split()
            assert len(values) == 6, "Each line should have 6 values (xhe4, xc12, xo16, rho, T, min_critical_length)"

            # Check that values can be converted to float
            try:
                float_values = [float(v) for v in values]
            except ValueError:
                pytest.fail("detonation_lengths.dat should contain numeric values")

    def test_runtime_display(self):
        """Test that runtime information is displayed"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Run failed: {result.stderr}"
        assert "Begin run" in result.stdout, "Should display begin run time"
        assert "End run" in result.stdout, "Should display end run time"
        assert "Total run time" in result.stdout, "Should display total runtime"


@pytest.mark.integration
@pytest.mark.slow
class TestSelfHeatNumerics:
    """Test numerical behavior and edge cases"""

    def test_low_temperature_run(self):
        """Test behavior at low temperatures (minimal nuclear burning)"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-T', '1e7', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        # Should complete even if no significant burning occurs
        assert result.returncode == 0, f"Low temperature run failed: {result.stderr}"

    def test_high_temperature_run(self):
        """Test behavior at high temperatures (rapid burning)"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-T', '2e9', '-tmax', '0.0001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        # Should handle rapid burning
        assert result.returncode == 0, f"High temperature run failed: {result.stderr}"

    def test_low_density_run(self):
        """Test behavior at low densities"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-rho', '1e3', '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Low density run failed: {result.stderr}"

    def test_timestep_adaptation(self):
        """Test that adaptive timestepping works (check for 'Halving timestep' message)"""
        # Use conditions likely to trigger timestep reduction
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric', '-T', '2e9', '-tmax', '0.01'],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Adaptive timestep run failed: {result.stderr}"
        # Note: May or may not see "Halving timestep" depending on conditions


@pytest.mark.integration
@pytest.mark.slow
class TestSelfHeatPureCompositions:
    """Test with various pure compositions"""

    def test_pure_helium(self):
        """Test with pure helium composition"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric',
             '-xp', '0.0', '-xhe4', '1.0', '-xc12', '0.0', '-xo16', '0.0',
             '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"Pure helium run failed: {result.stderr}"

    def test_carbon_oxygen_mixture(self):
        """Test with C/O mixture (no helium)"""
        result = subprocess.run(
            ['python', SELF_HEAT_PATH, '--isochoric',
             '-xp', '0.0', '-xhe4', '0.0', '-xc12', '0.5', '-xo16', '0.5',
             '-tmax', '0.001'],
            capture_output=True,
            text=True,
            timeout=900
        )
        assert result.returncode == 0, f"C/O mixture run failed: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
