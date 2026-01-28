# Testing Guide

## Setup

Install test dependencies:
```bash
uv add --dev pytest pytest-cov pytest-mock
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=lh_v2 --cov-report=html --cov-report=term
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only fast tests (exclude slow tests)
```bash
pytest -m "unit and not slow"
```

### Run tests for a specific module
```bash
pytest tests/unit/stats/
```

### Run a specific test file
```bash
pytest tests/unit/stats/test_correlations.py
```

### Run with verbose output
```bash
pytest -vv
```

### Using taskipy shortcuts
```bash
uv run task test              # Run all tests
uv run task test-cov          # Run with coverage
uv run task test-fast         # Run only fast unit tests
uv run task test-verbose      # Run with verbose output
```

## Test Organization

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Fast, isolated unit tests
│   ├── stats/
│   │   ├── test_correlations.py
│   │   └── test_norm_error.py
│   └── test_params.py
└── integration/             # Slower integration tests
    └── (coming soon)
```

## Writing Tests

### Basic test structure
```python
import pytest
from lh_v2.stats.correlations import pearson_correlation

@pytest.mark.unit
@pytest.mark.stats
def test_pearson_correlation():
    """Test description."""
    # Arrange
    ts1 = [1.0, 2.0, 3.0]
    ts2 = [1.0, 2.0, 3.0]
    
    # Act
    result = pearson_correlation(ts1, ts2)
    
    # Assert
    assert result == 1.0
```

### Using fixtures
```python
def test_with_fixture(sample_array):
    """Fixtures are defined in conftest.py."""
    assert len(sample_array) == 5
```

### Testing exceptions
```python
def test_raises_error():
    """Test that an error is raised."""
    with pytest.raises(ValueError):
        some_function_that_should_fail()
```

## Test Markers

Use markers to categorize tests:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Tests that take a long time
- `@pytest.mark.data_loading` - Data loading tests
- `@pytest.mark.driver_analysis` - Driver analysis tests
- `@pytest.mark.forecasting` - Forecasting tests
- `@pytest.mark.stats` - Statistical function tests

## Coverage Reports

After running tests with coverage:
- View HTML report: `open htmlcov/index.html`
- Coverage is also printed to terminal
- XML report for CI/CD: `coverage.xml`

## Best Practices

1. **One concept per test** - Each test should verify one behavior
2. **Descriptive names** - Test names should describe what they test
3. **AAA pattern** - Arrange, Act, Assert
4. **Use fixtures** - Reuse common setup via fixtures
5. **Mark tests** - Use markers to categorize tests
6. **Test edge cases** - Empty inputs, None, negative numbers, etc.
7. **Fast tests** - Keep unit tests fast (< 1 second each)

## Continuous Integration

Tests should be run on every commit via CI/CD (GitHub Actions).
See `.github/workflows/ci.yml` for configuration.
