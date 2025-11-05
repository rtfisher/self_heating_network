# Testing Documentation

This document describes the comprehensive test suite for the PYNUCDET (PYNUCastro Detonation Estimation Tool) codebase.

## Test Suite Overview

The test suite includes:

1. **Unit Tests** - Testing individual functions and modules
2. **Integration Tests** - Testing complete workflows and interactions
3. **Regression Tests** - Ensuring outputs match expected values

## Test Files

### test_aux.py
Tests for the `aux.py` auxiliary module:
- `TestHelmholtzCall`: Tests for Helmholtz EOS wrapper
  - Forward calls (isochoric)
  - Inverse calls (isobaric)
  - Different compositions
  - Temperature variations
  - Error handling
- `TestFloatToLatexScientific`: Tests for LaTeX formatting utility
  - Zero values
  - Scientific notation
  - Plain numbers
  - Precision control
  - Edge cases
- `TestHelmholtzWithTestData`: Regression tests against known test data

### test_cleanup.py
Tests for the `cleanup.py` module:
- File cleanup functionality
- Pattern matching for reaction flow files
- Preservation of non-target files
- Error handling for missing files

### test_self_heat.py
Tests for Helmholtz EOS integration:
- Basic forward and inverse calls
- Numerical accuracy tests
- Test data validation

### test_self_heat_integration.py
Integration tests for complete workflows:
- `TestSelfHeatIntegration`: End-to-end workflow tests
  - Isochoric runs
  - Isobaric runs
  - Custom parameters
  - Output file generation
  - Invalid input handling
- `TestSelfHeatNumerics`: Numerical behavior tests
  - Low/high temperature regimes
  - Low/high density regimes
  - Adaptive timestepping
- `TestSelfHeatPureCompositions`: Tests with various compositions
  - Pure helium
  - Carbon-oxygen mixtures
  - Custom abundance ratios

## Test Configuration

### conftest.py
Shared pytest fixtures:
- `helmholtz_executable`: Verifies Helmholtz executable exists
- `test_data_files`: Verifies test data files exist
- `clean_test_outputs`: Cleans up output files after tests
- `temp_working_directory`: Provides isolated temporary directory

### pytest.ini
Pytest configuration:
- Test discovery patterns
- Custom markers
- Coverage options
- Output formatting

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest test_aux.py
```

### Run Specific Test Class
```bash
pytest test_aux.py::TestHelmholtzCall
```

### Run Specific Test Function
```bash
pytest test_aux.py::TestHelmholtzCall::test_helmholtz_forward_call
```

### Run with Verbose Output
```bash
pytest -v
```

### Run with Coverage Report
```bash
pytest --cov=. --cov-report=html
```

### Run Only Fast Tests (Exclude Slow Tests)
```bash
pytest -m "not slow"
```

### Run Only Unit Tests
```bash
pytest -m unit
```

### Run Only Integration Tests
```bash
pytest -m integration
```

### Run Tests in Parallel
```bash
pytest -n auto
```

## Test Markers

Tests are marked with the following categories:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.helmholtz` - Tests requiring Helmholtz executable

## Continuous Integration

The test suite runs automatically via GitHub Actions on:
- Every push to main branch
- Every pull request to main
- Daily scheduled runs (00:00 UTC)
- Manual workflow dispatch

### CI Matrix Testing
Tests run on multiple Python versions:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

## Test Coverage

To generate and view coverage report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

Coverage reports are automatically uploaded to Codecov in CI.

## Prerequisites

Before running tests, ensure:

1. **Helmholtz Fortran code is compiled:**
   ```bash
   cd _helmholtz
   make
   cd ..
   ```

2. **Python dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-timeout pytest-xdist
   ```

3. **Test data files exist:**
   - `helm_input_test_data.txt`
   - `helm_output_test_data.txt`
   - `helm_table.dat`

## Writing New Tests

When adding new tests:

1. **Use descriptive names**: Test function names should clearly describe what they test
2. **Add docstrings**: Include brief descriptions of what each test validates
3. **Use fixtures**: Utilize shared fixtures from conftest.py
4. **Add markers**: Mark tests appropriately (unit/integration/slow)
5. **Test edge cases**: Include boundary conditions and error cases
6. **Keep tests isolated**: Tests should not depend on each other

### Example Test Structure
```python
import pytest

class TestNewFeature:
    """Test suite for new feature"""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality of new feature"""
        # Arrange
        input_data = setup_test_data()

        # Act
        result = new_feature(input_data)

        # Assert
        assert result == expected_value

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_workflow(self):
        """Test complete workflow with new feature"""
        # Test implementation
        pass
```

## Troubleshooting

### Tests Fail with "Helmholtz executable not found"
Compile the Helmholtz code:
```bash
cd _helmholtz && make && cd ..
```

### Tests Fail with Import Errors
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Integration Tests Time Out
Increase timeout or skip slow tests:
```bash
pytest -m "not slow"
```

### Coverage Reports Not Generated
Install pytest-cov:
```bash
pip install pytest-cov
```

## Contributing

When contributing code:

1. Write tests for new features
2. Ensure all tests pass locally before submitting PR
3. Maintain or improve code coverage
4. Update this documentation if adding new test categories

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
