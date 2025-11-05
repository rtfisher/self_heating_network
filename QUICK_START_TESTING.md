# Quick Start Guide - Testing

## Installation

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-timeout pytest-xdist
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run only fast tests (exclude slow integration tests):
```bash
pytest -m "not slow"
```

### Run only unit tests:
```bash
pytest -m unit
```

### Run only integration tests:
```bash
pytest -m integration
```

### Run with coverage report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # View coverage in browser
```

### Run in parallel (faster):
```bash
pytest -n auto
```

### Run specific test file:
```bash
pytest test_aux.py
```

### Run with verbose output:
```bash
pytest -v
```

## Test Files

| File | Description | Test Count |
|------|-------------|------------|
| `test_aux.py` | Tests for aux.py module (Helmholtz, LaTeX) | 15+ |
| `test_cleanup.py` | Tests for cleanup.py functionality | 8 |
| `test_self_heat.py` | Tests for Helmholtz EOS integration | 5 |
| `test_self_heat_integration.py` | End-to-end workflow tests | 20+ |

## Prerequisites

Before running tests:
```bash
cd _helmholtz
make
cd ..
```

## Common Issues

### "Helmholtz executable not found"
**Solution:** Compile Helmholtz first
```bash
cd _helmholtz && make && cd ..
```

### "Test data files not found"
**Solution:** Ensure you're in the project root directory with:
- `helm_input_test_data.txt`
- `helm_output_test_data.txt`
- `helm_table.dat`

### Tests timeout
**Solution:** Run without slow tests
```bash
pytest -m "not slow"
```

## CI/CD

Tests run automatically on GitHub Actions:
- Every push to main
- Every pull request
- Daily at midnight UTC
- Can be triggered manually

View results at: https://github.com/[your-repo]/actions

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.helmholtz` - Requires Helmholtz executable

## Quick Commands Cheat Sheet

```bash
# Install dependencies
pip install pytest pytest-cov pytest-timeout pytest-xdist

# Compile Helmholtz
cd _helmholtz && make && cd ..

# Run fast tests only
pytest -m "not slow"

# Run with coverage
pytest --cov=. --cov-report=html

# Run in parallel
pytest -n auto

# Run specific test
pytest test_aux.py::TestHelmholtzCall::test_helmholtz_forward_call

# List all tests without running
pytest --collect-only
```

## Getting Help

For detailed documentation, see `TESTING.md`

For test suite summary, see `TEST_SUITE_SUMMARY.md`
