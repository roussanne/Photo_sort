# Unified Sort Testing Suite

Comprehensive automated tests for the Photo_sort/Unified Sort project.

## Test Structure

```
tests/
├── conftest.py           # Pytest fixtures and configuration
├── test_core.py          # Core analysis function tests
├── test_auto_sort.py     # Auto-classification tests
└── README.md             # This file
```

## Running Tests

### Install Testing Dependencies

```bash
# Install pytest
pip install pytest

# Optional: Install additional testing tools
pip install pytest-cov     # Coverage reporting
pip install pytest-xdist   # Parallel execution
pip install pytest-timeout # Test timeouts
```

### Run All Tests

```bash
# From the unified-sort/ directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=unified_sort --cov-report=html
```

### Run Specific Test Files

```bash
# Run only core tests
pytest tests/test_core.py

# Run only auto-sort tests
pytest tests/test_auto_sort.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_core.py::TestBatchAnalyze

# Run a specific test function
pytest tests/test_core.py::TestBatchAnalyze::test_batch_analyze_simple_mode
```

### Run Tests by Marker

```bash
# Skip slow tests
pytest -m "not slow"

# Run only tests requiring PyTorch
pytest -m torch_required

# Run integration tests only
pytest -m integration
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Test Categories

### Unit Tests

- **test_core.py**: Tests for core analysis functions
  - Image loading and discovery
  - Thumbnail generation
  - Batch analysis (simple and advanced modes)
  - Multiprocessing support
  - Helper functions (pHash, hamming distance, etc.)

- **test_auto_sort.py**: Tests for auto-classification
  - Configuration management
  - Confidence-based classification
  - Batch classification
  - Adaptive thresholds
  - Statistical analysis
  - Config recommendations

### Fixtures (conftest.py)

- **temp_test_dir**: Session-scoped temporary directory
- **sample_sharp_image**: Sharp test image (checkerboard pattern)
- **sample_defocus_image**: Defocus blur test image (Gaussian blur)
- **sample_motion_blur_image**: Motion blur test image (horizontal blur)
- **sample_images**: Dictionary of all sample images
- **sample_image_batch**: Batch of varied quality images
- **mock_exif_data**: Mock EXIF metadata
- **mock_high_iso_exif**: Mock high-ISO EXIF data
- **sample_scores**: Sample classification scores
- **sample_uncertain_scores**: Uncertain classification scores
- **sample_low_quality_scores**: Low quality scores

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
import unified_sort as us

class TestMyFeature:
    """Test suite for my new feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        result = us.my_new_function()
        assert result is not None

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            us.my_new_function(invalid_input)

    @pytest.mark.slow
    def test_performance(self):
        """Test performance with large dataset."""
        # Slow test marked for optional skipping
        pass
```

### Using Fixtures

```python
def test_with_sample_image(sample_sharp_image):
    """Test using a fixture."""
    result = us.analyze_image(str(sample_sharp_image))
    assert result["sharp_score"] > 0.5
```

## Test Coverage

Generate coverage report:

```bash
pytest --cov=unified_sort --cov-report=html
```

View coverage in browser:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

Target coverage: **>80%** for core modules

## Continuous Integration

### GitHub Actions (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest --cov
```

## Known Issues

- Some tests require sample images and may take a few seconds to run
- Multiprocessing tests may be slower on Windows
- PyTorch-dependent tests will be skipped if PyTorch is not installed
- Google Drive tests require API credentials

## Troubleshooting

### "Module not found" errors

```bash
# Ensure package is installed in development mode
cd /path/to/unified-sort
pip install -e .
```

### Fixture errors

```bash
# Ensure you're running from the correct directory
cd /path/to/unified-sort
pytest tests/
```

### Slow tests

```bash
# Skip slow tests
pytest -m "not slow"
```

## Future Test Additions

Planned test coverage:

- [ ] EXIF module tests (test_exif_adjust.py)
- [ ] Face detection tests (test_detection.py)
- [ ] Deep learning NR-IQA tests (test_nn_iqa.py)
- [ ] Google Drive integration tests (test_gdrive.py)
- [ ] Pipeline tests (test_pipeline.py)
- [ ] I/O utilities tests (test_io_utils.py)
- [ ] Streamlit UI tests (requires selenium)
- [ ] End-to-end integration tests

## Contributing

When adding new features:

1. Write tests for new functionality
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov`
4. Add appropriate markers (@pytest.mark.slow, etc.)
5. Update this README if adding new test files

---

*Last Updated: 2025-11-17*
*Testing Framework: pytest ≥7.0*
