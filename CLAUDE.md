# CLAUDE.md - AI Assistant Guide for Photo_sort

> Comprehensive guide for AI assistants working on the Unified Image Quality Classifier project

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Core Architecture](#core-architecture)
- [Development Workflows](#development-workflows)
- [Key Conventions](#key-conventions)
- [Common Tasks](#common-tasks)
- [Testing Strategy](#testing-strategy)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is Photo_sort?

Photo_sort (Unified Image Quality Classifier) is a **Streamlit-based image quality analysis tool** that automatically classifies photos as:
- **Sharp** (선명 ✅): Clear, well-focused images
- **Defocus** (아웃포커스 🌫️): Out-of-focus blur
- **Motion** (모션블러 📸): Motion blur from camera/subject movement

### Key Features

1. **Dual-Mode Analysis**
   - Simple Mode: Fast Laplacian-based blur detection
   - Advanced Mode: Multi-feature analysis with 7+ image quality metrics

2. **Confidence-Based Auto-Sorting**
   - Automatic classification with confidence scores
   - Uncertainty detection for manual review
   - Adaptive thresholds based on dataset statistics

3. **Hybrid Pipeline**
   - EXIF metadata integration
   - Face detection weighting
   - Deep learning NR-IQA (No-Reference Image Quality Assessment)
   - ROI-free subject-agnostic analysis

4. **Additional Features**
   - pHash-based duplicate detection
   - HEIC/HEIF support (iPhone photos)
   - RAW image processing (RW2 → JPG)
   - Safe deletion (trash instead of permanent delete)
   - CSV export and dataset generation for ML training

### Technology Stack

- **UI Framework**: Streamlit ≥1.36
- **Image Processing**: OpenCV, NumPy, Pillow
- **Data Handling**: Pandas
- **Optional**: PyTorch (deep learning), rawpy (RAW processing), pillow-heif (HEIC support)
- **Language**: Python ≥3.9

---

## Repository Structure

### Directory Layout

```
Photo_sort/
├── readme.md                     # Main documentation (Korean)
├── CLAUDE.md                     # This file - AI assistant guide
│
├── unified-sort/                 # MAIN ACTIVE CODEBASE
│   ├── src/unified_sort/        # Core package
│   │   ├── __init__.py          # Package entry point, API exports
│   │   ├── core.py              # Core analysis functions
│   │   ├── helpers.py           # Utility functions (pHash, etc.)
│   │   ├── io_utils.py          # I/O operations (imread, export)
│   │   ├── pipeline.py          # Hybrid analysis pipeline
│   │   └── auto_sort.py         # NEW: Confidence-based classification
│   │
│   ├── app/
│   │   └── streamlit_app.py     # Streamlit UI application
│   │
│   ├── pyproject.toml           # Package metadata (setuptools)
│   ├── setup.cfg                # Build configuration
│   ├── requirements.txt         # Python dependencies
│   └── LICENSE                  # MIT License
│
├── old/                          # LEGACY CODE - DO NOT MODIFY
│   ├── unified-sort_legacy/     # Previous version
│   └── *.py                     # Old standalone scripts
│
└── imgsim/                       # Image similarity utilities
    ├── imgsim.py                # Standalone similarity tool
    └── asdf.md                  # Documentation
```

### Key Files Reference

| File Path | Purpose | When to Modify |
|-----------|---------|----------------|
| `unified-sort/src/unified_sort/__init__.py` | Public API, module exports | Adding new public functions |
| `unified-sort/src/unified_sort/core.py` | Core analysis logic | Changing analysis algorithms |
| `unified-sort/src/unified_sort/auto_sort.py` | Auto-classification system | Tuning classification behavior |
| `unified-sort/src/unified_sort/pipeline.py` | Hybrid pipeline | Adding advanced features |
| `unified-sort/app/streamlit_app.py` | UI interface | UI changes, new modes |
| `readme.md` | User documentation | Feature updates |

---

## Core Architecture

### Module Responsibilities

#### 1. `core.py` - Core Analysis Engine
**Purpose**: Fundamental image quality assessment

**Key Functions**:
- `list_images(root, recursive)`: File discovery
- `load_thumbnail(path, max_side)`: Memory-efficient image loading
- `batch_analyze(paths, mode, ...)`: Simple/advanced batch analysis
- `compute_scores_advanced(gray, tiles, params)`: Multi-feature extraction

**Analysis Modes**:
- **Simple**: Laplacian variance + Sobel edges + directional analysis
- **Advanced**: 7-feature extraction (VoL, Tenengrad, HFR, ESW, RSS, AI, STR) - **TODO: Not fully implemented**

#### 2. `auto_sort.py` - Confidence-Based Classification
**Purpose**: Intelligent auto-labeling with uncertainty detection

**Key Classes**:
- `AutoSortConfig`: Configurable thresholds and strategies
- `ClassificationResult`: Rich classification result with confidence metrics

**Key Functions**:
- `classify_with_confidence(scores, config)`: Multi-stage decision logic
- `batch_classify(scores_dict, config)`: Batch classification
- `compute_adaptive_thresholds(scores_dict)`: Dataset-aware thresholds
- `get_classification_stats(results)`: Statistical analysis
- `suggest_config_adjustments(stats)`: Smart recommendations

**Decision Pipeline**:
1. Apply user bias to scores
2. Calculate confidence (margin between top 2 classes)
3. Validate total quality (detect analysis failures)
4. Check class-specific thresholds
5. Detect uncertainty (narrow margins)
6. Assign label or flag for manual review

#### 3. `pipeline.py` - Hybrid Analysis Pipeline
**Purpose**: Advanced multi-strategy analysis

**Key Functions**:
- `analyze_one_full_hybrid(path, params)`: Single-image hybrid analysis
- `batch_analyze_full_hybrid(paths, params, max_workers)`: Parallel hybrid processing
- `unload_dl_model()`: Memory management for DL models

**Hybrid Features** (Partially Implemented):
- EXIF correction (ISO, shutter speed, focal length)
- Face detection weighting
- Deep learning fusion
- ROI-free analysis

#### 4. `io_utils.py` - I/O Utilities
**Purpose**: Format-agnostic image loading and dataset export

**Key Functions**:
- `imread_any(path)`: Universal image reader (HEIC, RAW, standard formats)
- `export_labeled_dataset(labels, out_root, move)`: ML dataset generation

#### 5. `helpers.py` - Helper Utilities
**Purpose**: Common utility functions

**Key Functions**:
- `load_fullres(path)`: Load full-resolution image
- `phash_from_gray(gray)`: Perceptual hash calculation
- `hamming_dist(hash1, hash2)`: Hash distance for similarity
- `make_widget_key(...)`: Streamlit widget key generation

---

## Development Workflows

### Environment Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd Photo_sort

# 2. Install in development mode
cd unified-sort
pip install -e .

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Install optional dependencies as needed
pip install pillow-heif rawpy imageio send2trash torch torchvision

# 5. Verify installation
python -c "import unified_sort as us; us.print_status()"
```

### Running the Application

```bash
# From repository root
cd unified-sort
streamlit run app/streamlit_app.py

# Or from anywhere after installation
streamlit run <path-to-repo>/unified-sort/app/streamlit_app.py
```

### Git Workflow

**Current Branch Structure**:
- Working branch: `claude/claude-md-mi2lcgyvulc9ed28-01RsZ326RVEaK9yv3dyzYNYU`
- Previous feature branch: `claude/testing-mi2ks8b93cydeo29-01F7G3uJFY2K4Bhf4ogpv9W2`

**Branch Naming Convention**: `claude/<description>-<session-id>`

**Commit Message Style** (from git log):
- Use descriptive prefixes: `feat:`, `fix:`, `docs:`, etc.
- Example: `feat: Add advanced auto-sorting with confidence-based classification`
- Keep messages concise but informative

**Push Protocol**:
```bash
# Always use -u flag for new branches
git push -u origin <branch-name>

# Branch must start with 'claude/' and match session ID
# Retry up to 4 times with exponential backoff on network errors: 2s, 4s, 8s, 16s
```

---

## Key Conventions

### 1. Code Organization

**Package Structure**:
- All main code lives in `unified-sort/src/unified_sort/`
- Streamlit app lives in `unified-sort/app/`
- Legacy code in `old/` should **NEVER** be modified

**Import Pattern**:
```python
# In __init__.py - graceful degradation
try:
    from .core import list_images, batch_analyze
    _CORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Core module import failed: {e}")
    _CORE_AVAILABLE = False
    def list_images(*args, **kwargs):
        raise NotImplementedError("Core module not available")
```

### 2. Error Handling

**Pattern**: Specific exceptions with user-friendly fallbacks
```python
# Good - Specific exception handling
try:
    img = imread_any(path)
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Failed to import: {e}")
    return None
except (OSError, PermissionError) as e:
    print(f"Warning: File access error for {path}: {e}")
    return None

# Bad - Catching generic Exception
try:
    img = imread_any(path)
except Exception:
    return None
```

**Principles**:
- Use specific exception types
- Provide helpful error messages
- Include context (file path, operation) in messages
- Return sensible defaults instead of crashing

### 3. Type Hints

**Standard**: All public functions must have type hints
```python
def batch_analyze(
    paths: List[str],
    mode: str = "simple",
    tiles: int = 4,
    params: Optional[dict] = None,
    max_workers: int = 1
) -> Dict[str, dict]:
    """Docstring..."""
    pass
```

### 4. Documentation

**Style**: Google-style docstrings
```python
def classify_with_confidence(
    scores: Dict[str, float],
    config: AutoSortConfig
) -> ClassificationResult:
    """
    점수를 분석하여 신뢰도 기반 분류를 수행합니다.

    이 함수는 단순 argmax가 아닌 다층 결정 로직을 사용합니다:
    1. 바이어스 적용
    2. 최고 점수 클래스 선택
    3. 최소 임계값 검증
    4. 마진/신뢰도 검증
    5. 전체 품질 게이팅
    6. 불확실성 감지

    Args:
        scores: 원본 점수 딕셔너리
        config: 설정 객체

    Returns:
        ClassificationResult 객체
    """
```

**Bilingual Notes**: Comments may be in Korean or English
- User-facing docs (readme.md): Korean
- Code comments: Mixed (prefer English for AI readability)
- This file (CLAUDE.md): English

### 5. Configuration

**Pattern**: Dataclass-based configuration
```python
from dataclasses import dataclass

@dataclass
class AutoSortConfig:
    """Configuration with sensible defaults"""
    min_sharp: float = 0.35
    min_defocus: float = 0.35
    min_motion: float = 0.35
    strategy: Literal["conservative", "balanced", "aggressive"] = "balanced"
```

### 6. Validation

**Always validate inputs**:
```python
def load_thumbnail(path: str, max_side: int = 384) -> Optional[np.ndarray]:
    # Type validation
    if not isinstance(max_side, int) or max_side <= 0:
        max_side = 384  # Reset to default

    # File existence check
    if not Path(path).exists():
        return None

    # Image validity check
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
```

---

## Common Tasks

### Adding a New Analysis Feature

1. **Add core function** to `core.py` or `pipeline.py`
2. **Export** in `__init__.py` under appropriate section
3. **Add to `__all__`** list
4. **Update UI** in `streamlit_app.py` if needed
5. **Update docs** in `readme.md`

Example:
```python
# In core.py
def compute_new_metric(gray: np.ndarray) -> float:
    """Compute new quality metric."""
    # Implementation
    pass

# In __init__.py
from .core import compute_new_metric  # Add import
__all__ = [..., "compute_new_metric"]  # Add to exports
```

### Modifying Classification Behavior

**File**: `auto_sort.py`

**Common Adjustments**:
- Change default thresholds in `AutoSortConfig.__init__()`
- Modify strategy presets in `AutoSortConfig._apply_strategy()`
- Adjust decision logic in `classify_with_confidence()`
- Update statistical analysis in `get_classification_stats()`

**Testing Changes**:
```python
import unified_sort as us

# Create test config
config = us.AutoSortConfig(
    min_confidence=0.20,
    strategy="conservative"
)

# Test classification
scores = {"sharp_score": 0.6, "defocus_score": 0.3, "motion_score": 0.1}
result = us.classify_with_confidence(scores, config)

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Needs Review: {result.needs_review}")
```

### Adding New Image Format Support

**File**: `io_utils.py`

**Pattern**:
```python
def imread_any(path: str) -> Optional[np.ndarray]:
    ext = Path(path).suffix.lower()

    # Add new format handler
    if ext in ['.new_format']:
        try:
            import new_format_lib
            img = new_format_lib.load(path)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except ImportError:
            print("Warning: new_format_lib not installed")
            return None

    # Existing handlers...
```

### Updating UI

**File**: `unified-sort/app/streamlit_app.py`

**Key Sections**:
- Mode selection (Simple/Advanced/Hybrid)
- Parameter controls (sliders, checkboxes)
- Result display (dataframes, images, metrics)
- Export functionality (CSV, dataset)

**Streamlit Patterns**:
```python
# Session state for persistence
if "analyzed_results" not in st.session_state:
    st.session_state["analyzed_results"] = {}

# Widget keys for uniqueness
key = us.make_widget_key("threshold", "sharp", index)

# Conditional rendering based on mode
mode = st.radio("모드 선택", ["간단", "고급", "하이브리드"])
if mode == "간단":
    # Simple mode UI
elif mode == "고급":
    # Advanced mode UI
```

---

## Testing Strategy

### Manual Testing Checklist

Since there are no automated tests yet, use this checklist:

**Installation Test**:
```bash
cd unified-sort
pip install -e .
python -c "import unified_sort as us; us.print_status()"
# All core modules should show ✓
```

**Core Functionality**:
```python
import unified_sort as us

# 1. Test image listing
paths = us.list_images("test_images/", recursive=True)
assert len(paths) > 0, "No images found"

# 2. Test simple analysis
results = us.batch_analyze(paths, mode="simple")
assert len(results) > 0, "Analysis failed"

# 3. Test classification
config = us.AutoSortConfig()
classifications = us.batch_classify(results, config)
stats = us.get_classification_stats(classifications)
print(stats)
```

**UI Test**:
```bash
streamlit run app/streamlit_app.py
# Manual checks:
# - Load folder with images
# - Run simple mode analysis
# - Check results display correctly
# - Export CSV
# - Try advanced mode (if implemented)
```

### Test Data Setup

**Recommended Test Dataset**:
- 10-20 images of varying quality
- Mix of sharp, blurry, and motion-blurred images
- Include edge cases: very small, very large, different formats
- Test optional formats: HEIC (if available), RAW (if available)

### Validation After Changes

After modifying code, always:
1. Check package imports: `python -c "import unified_sort"`
2. Run basic analysis on test images
3. Verify UI still loads without errors
4. Check that exports work (CSV, dataset)

---

## Troubleshooting

### Common Issues

#### 1. "Core module is not available"
**Symptom**: Warning on import
**Cause**: Package not installed or circular import
**Fix**:
```bash
cd unified-sort
pip install -e .
```

#### 2. HEIC/RAW Images Not Loading
**Symptom**: Images return None
**Cause**: Optional dependencies not installed
**Fix**:
```bash
pip install pillow-heif  # For HEIC
pip install rawpy imageio  # For RAW
```

#### 3. Streamlit Session State Issues
**Symptom**: UI state resets unexpectedly
**Cause**: Widget key collisions
**Fix**: Use `make_widget_key()` for unique keys
```python
key = us.make_widget_key("slider", "threshold", img_path, index)
st.slider("Threshold", key=key)
```

#### 4. Memory Issues with Large Batches
**Symptom**: OOM errors during batch processing
**Fix**: Use thumbnails and process in smaller chunks
```python
# Don't load all full-res at once
for path in paths:
    thumb = us.load_thumbnail(path, max_side=512)  # Smaller
    # Process thumb
```

#### 5. Import Errors in Development
**Symptom**: `ModuleNotFoundError: No module named 'unified_sort'`
**Cause**: Not in development mode or wrong directory
**Fix**:
```bash
# Ensure you're in unified-sort/ directory
cd unified-sort
pip install -e .

# Or use absolute imports in tests
import sys
sys.path.insert(0, '/path/to/Photo_sort/unified-sort/src')
```

---

## Development Roadmap

### Current Status (v0.1.0)

**Completed**:
- ✅ Core analysis engine (simple mode)
- ✅ Streamlit UI (basic + advanced mode)
- ✅ Auto-classification with confidence scoring
- ✅ EXIF metadata integration
- ✅ Thread-safe model management
- ✅ Error handling improvements
- ✅ Type hints and documentation

**In Progress** (Partially Implemented):
- 🔄 7-feature advanced analysis (placeholder in core.py)
- 🔄 Face detection module
- 🔄 Deep learning NR-IQA integration
- 🔄 Hybrid pipeline (structure exists, features incomplete)

**Planned**:
- 📋 3-class CNN model training
- 📋 Cloud storage integration (Google Drive, Dropbox)
- 📋 Profile system (save/load configurations)
- 📋 REST API
- 📋 CLI tool
- 📋 Automated testing suite

---

## Best Practices for AI Assistants

### When Working on This Codebase

1. **Always work in `unified-sort/` directory** - Never modify `old/`
2. **Check installation status first**: `us.print_status()`
3. **Use type hints**: All new functions must have type annotations
4. **Handle errors gracefully**: Specific exceptions with helpful messages
5. **Validate inputs**: Check types, ranges, file existence
6. **Update `__all__`**: When adding public functions
7. **Test before committing**: Manual testing checklist (no automated tests yet)
8. **Update documentation**: Both docstrings and readme.md
9. **Follow Korean naming conventions**: UI text and user-facing messages in Korean
10. **Use graceful degradation**: Optional features should fail softly

### Code Review Checklist

Before considering code complete:
- [ ] Type hints added to all function signatures
- [ ] Google-style docstrings added
- [ ] Input validation implemented
- [ ] Error handling with specific exceptions
- [ ] Exported in `__init__.py` if public API
- [ ] Manual testing completed
- [ ] readme.md updated if user-facing feature
- [ ] No modifications to `old/` directory
- [ ] Git commit message follows convention

---

## Package API Reference

### Public API (`import unified_sort as us`)

**Core Functions**:
- `us.list_images(root, recursive=False)` → List[str]
- `us.batch_analyze(paths, mode="simple", ...)` → Dict[str, dict]
- `us.load_thumbnail(path, max_side=384)` → Optional[np.ndarray]

**I/O Functions**:
- `us.imread_any(path)` → Optional[np.ndarray]
- `us.export_labeled_dataset(labels, out_root, move=False)` → Tuple[int, Path]

**Helper Functions**:
- `us.phash_from_gray(gray)` → int
- `us.hamming_dist(hash1, hash2)` → int
- `us.load_fullres(path)` → Optional[np.ndarray]

**Classification**:
- `us.AutoSortConfig(...)` → AutoSortConfig
- `us.classify_with_confidence(scores, config)` → ClassificationResult
- `us.batch_classify(scores_dict, config)` → Dict[str, ClassificationResult]
- `us.compute_adaptive_thresholds(scores_dict)` → Dict[str, float]
- `us.get_classification_stats(results)` → Dict
- `us.suggest_config_adjustments(stats)` → List[str]

**Hybrid Pipeline** (Advanced):
- `us.analyze_one_full_hybrid(path, params)` → Dict[str, float]
- `us.batch_analyze_full_hybrid(paths, params, max_workers=8)` → Dict[str, Dict]
- `us.unload_dl_model()` → None

**Utility**:
- `us.check_installation()` → Dict[str, bool]
- `us.print_status()` → None
- `us.get_version()` → str

---

## Contact & Resources

**Repository**: Check git remote for URL
**License**: MIT License (see unified-sort/LICENSE)
**Python Version**: ≥3.9
**Main Dependencies**: Streamlit, OpenCV, NumPy, Pandas, Pillow

**For Issues**: Check error messages, consult troubleshooting section, verify installation status

---

*Last Updated: 2025-11-17 (v0.1.0)*
*This guide is maintained for AI assistants working on the Photo_sort project.*
