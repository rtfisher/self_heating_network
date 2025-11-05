"""
Pytest configuration file with shared fixtures and setup
"""
import pytest
import os
import tempfile
import shutil


@pytest.fixture(scope="session")
def helmholtz_executable():
    """
    Session-scoped fixture to verify Helmholtz executable exists
    """
    exe_path = os.path.join("_helmholtz", "helmholtz.exe")
    if not os.path.exists(exe_path):
        pytest.skip(f"Helmholtz executable not found at {exe_path}. Run 'make' in _helmholtz directory.")
    return exe_path


@pytest.fixture(scope="session")
def test_data_files():
    """
    Session-scoped fixture to verify test data files exist
    """
    input_file = "helm_input_test_data.txt"
    output_file = "helm_output_test_data.txt"

    if not os.path.exists(input_file):
        pytest.skip(f"Test input data file not found: {input_file}")
    if not os.path.exists(output_file):
        pytest.skip(f"Test output data file not found: {output_file}")

    return input_file, output_file


@pytest.fixture
def clean_test_outputs():
    """
    Fixture to clean up test output files after each test
    """
    yield
    # Cleanup after test
    output_files = [
        "abundances.png",
        "detonation_lengths.png",
        "nuclear_network.py",
        "detonation_lengths.dat"
    ]
    for filename in output_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass

    # Clean up reaction flow files
    import glob
    for filename in glob.glob("reaction_flow_*.png"):
        try:
            os.remove(filename)
        except Exception:
            pass


@pytest.fixture
def temp_working_directory():
    """
    Fixture to create and use a temporary working directory
    """
    original_dir = os.getcwd()
    temp_dir = tempfile.mkdtemp()

    # Copy necessary files to temp directory
    files_to_copy = [
        "helm_input_test_data.txt",
        "helm_output_test_data.txt",
        "helm_table.dat"
    ]

    for filename in files_to_copy:
        if os.path.exists(filename):
            shutil.copy2(filename, temp_dir)

    # Copy _helmholtz directory
    helmholtz_src = "_helmholtz"
    helmholtz_dst = os.path.join(temp_dir, "_helmholtz")
    if os.path.exists(helmholtz_src):
        shutil.copytree(helmholtz_src, helmholtz_dst)

    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)


def pytest_configure(config):
    """
    Configure pytest with custom markers
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "helmholtz: marks tests that require Helmholtz executable"
    )
