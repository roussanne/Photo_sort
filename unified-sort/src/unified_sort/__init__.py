"""
unified_sort package

이미지 품질 분류를 위한 통합 패키지입니다.
간단한 라플라시안 기반 분석부터 고급 하이브리드 파이프라인까지 제공합니다.

주요 기능:
- 이미지 품질 분석 (선명도, 아웃포커스, 모션블러)
- EXIF 메타데이터 활용
- 얼굴 검출 기반 가중치
- 딥러닝 NR-IQA 통합
- 썸네일 생성 및 캐싱
- pHash 기반 중복 검출
- 학습 데이터셋 내보내기

사용 예시:
    import unified_sort as us
    
    # 이미지 목록 가져오기
    paths = us.list_images("/path/to/images", recursive=True)
    
    # 간단 분석
    results = us.batch_analyze(paths, mode="simple")
    
    # 하이브리드 분석
    params = {"exif_correction": True, "face_prior_enabled": True}
    results = us.batch_analyze_full_hybrid(paths, params=params)
"""

from typing import List

# 버전 정보
__version__ = "0.1.0"
__author__ = "Your Name"

# =====================================================================
# 핵심 분석 함수 임포트
# =====================================================================

try:
    from .core import (
        list_images,
        batch_analyze,
        load_thumbnail,
        compute_scores_advanced,
    )
    _CORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Core module import failed: {e}")
    _CORE_AVAILABLE = False
    # 폴백 스텁 (최소 기능)
    def list_images(*args, **kwargs):
        raise NotImplementedError("Core module not available")
    def batch_analyze(*args, **kwargs):
        raise NotImplementedError("Core module not available")
    def load_thumbnail(*args, **kwargs):
        raise NotImplementedError("Core module not available")
    def compute_scores_advanced(*args, **kwargs):
        raise NotImplementedError("Core module not available")

# =====================================================================
# I/O 유틸리티 임포트
# =====================================================================

try:
    from .io_utils import imread_any, export_labeled_dataset
    _IO_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: IO utils import failed: {e}")
    _IO_AVAILABLE = False
    def imread_any(*args, **kwargs):
        raise NotImplementedError("IO utils not available")
    def export_labeled_dataset(*args, **kwargs):
        raise NotImplementedError("IO utils not available")

# =====================================================================
# 헬퍼 함수 임포트
# =====================================================================

try:
    from .helpers import (
        load_fullres,
        phash_from_gray,
        hamming_dist,
        make_widget_key,
    )
    _HELPERS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Helpers import failed: {e}")
    _HELPERS_AVAILABLE = False
    def load_fullres(*args, **kwargs):
        raise NotImplementedError("Helpers not available")
    def phash_from_gray(*args, **kwargs):
        raise NotImplementedError("Helpers not available")
    def hamming_dist(*args, **kwargs):
        raise NotImplementedError("Helpers not available")
    def make_widget_key(*args, **kwargs):
        raise NotImplementedError("Helpers not available")

# =====================================================================
# 하이브리드 파이프라인 임포트
# =====================================================================

try:
    from .pipeline import (
        analyze_one_full_hybrid,
        batch_analyze_full_hybrid,
        unload_dl_model,
    )
    _PIPELINE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Pipeline import failed: {e}")
    _PIPELINE_AVAILABLE = False
    def analyze_one_full_hybrid(*args, **kwargs):
        raise NotImplementedError("Pipeline not available")
    def batch_analyze_full_hybrid(*args, **kwargs):
        raise NotImplementedError("Pipeline not available")
    def unload_dl_model(*args, **kwargs):
        raise NotImplementedError("Pipeline not available")

# =====================================================================
# 자동 분류 모듈 임포트
# =====================================================================

try:
    from .auto_sort import (
        AutoSortConfig,
        ClassificationResult,
        classify_with_confidence,
        batch_classify,
        compute_adaptive_thresholds,
        get_classification_stats,
        suggest_config_adjustments,
    )
    _AUTO_SORT_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Auto sort import failed: {e}")
    _AUTO_SORT_AVAILABLE = False
    def AutoSortConfig(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def ClassificationResult(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def classify_with_confidence(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def batch_classify(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def compute_adaptive_thresholds(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def get_classification_stats(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")
    def suggest_config_adjustments(*args, **kwargs):
        raise NotImplementedError("Auto sort not available")

# =====================================================================
# 얼굴 검출 모듈 임포트
# =====================================================================

try:
    from .detection import (
        detect_faces,
        compute_face_region_sharpness,
        apply_face_prior,
        get_face_coverage_ratio,
        visualize_face_detection,
    )
    _DETECTION_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Detection module import failed: {e}")
    _DETECTION_AVAILABLE = False
    def detect_faces(*args, **kwargs):
        raise NotImplementedError("Detection not available")
    def compute_face_region_sharpness(*args, **kwargs):
        raise NotImplementedError("Detection not available")
    def apply_face_prior(*args, **kwargs):
        raise NotImplementedError("Detection not available")
    def get_face_coverage_ratio(*args, **kwargs):
        raise NotImplementedError("Detection not available")
    def visualize_face_detection(*args, **kwargs):
        raise NotImplementedError("Detection not available")

# =====================================================================
# EXIF 메타데이터 모듈 임포트
# =====================================================================

try:
    from .exif_adjust import (
        extract_exif_data,
        compute_exif_adjustment_factors,
        apply_exif_adjustment,
        get_blur_risk_assessment,
    )
    _EXIF_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: EXIF module import failed: {e}")
    _EXIF_AVAILABLE = False
    def extract_exif_data(*args, **kwargs):
        raise NotImplementedError("EXIF module not available")
    def compute_exif_adjustment_factors(*args, **kwargs):
        raise NotImplementedError("EXIF module not available")
    def apply_exif_adjustment(*args, **kwargs):
        raise NotImplementedError("EXIF module not available")
    def get_blur_risk_assessment(*args, **kwargs):
        raise NotImplementedError("EXIF module not available")

# =====================================================================
# Deep Learning NR-IQA 모듈 임포트
# =====================================================================

try:
    from .nn_iqa import (
        NNQuality,
        get_model,
        unload_model,
        predict_quality,
        fuse_scores,
        is_available as nn_is_available,
        get_device_info,
    )
    _NN_IQA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: NN-IQA module import failed: {e}")
    _NN_IQA_AVAILABLE = False
    def NNQuality(*args, **kwargs):
        raise NotImplementedError("NN-IQA module not available")
    def get_model(*args, **kwargs):
        raise NotImplementedError("NN-IQA module not available")
    def unload_model(*args, **kwargs):
        raise NotImplementedError("NN-IQA module not available")
    def predict_quality(*args, **kwargs):
        raise NotImplementedError("NN-IQA module not available")
    def fuse_scores(*args, **kwargs):
        raise NotImplementedError("NN-IQA module not available")
    def nn_is_available(*args, **kwargs):
        return False
    def get_device_info(*args, **kwargs):
        return {"torch_available": False}

# =====================================================================
# Google Drive 통합 모듈 임포트
# =====================================================================

try:
    from .gdrive import (
        GDriveUploader,
        is_available as gdrive_is_available,
        setup_credentials,
        get_credentials_instructions,
    )
    _GDRIVE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"Warning: Google Drive module import failed: {e}")
    _GDRIVE_AVAILABLE = False
    def GDriveUploader(*args, **kwargs):
        raise NotImplementedError("Google Drive module not available")
    def gdrive_is_available(*args, **kwargs):
        return False
    def setup_credentials(*args, **kwargs):
        raise NotImplementedError("Google Drive module not available")
    def get_credentials_instructions(*args, **kwargs):
        return "Google Drive integration not available (dependencies not installed)"

# =====================================================================
# 공개 API 정의
# =====================================================================

__all__ = [
    # 버전 정보
    "__version__",

    # 핵심 함수
    "list_images",
    "batch_analyze",
    "load_thumbnail",
    "compute_scores_advanced",

    # I/O
    "imread_any",
    "export_labeled_dataset",

    # 헬퍼
    "load_fullres",
    "phash_from_gray",
    "hamming_dist",
    "make_widget_key",

    # 하이브리드 파이프라인
    "analyze_one_full_hybrid",
    "batch_analyze_full_hybrid",
    "unload_dl_model",

    # 자동 분류
    "AutoSortConfig",
    "ClassificationResult",
    "classify_with_confidence",
    "batch_classify",
    "compute_adaptive_thresholds",
    "get_classification_stats",
    "suggest_config_adjustments",

    # 얼굴 검출
    "detect_faces",
    "compute_face_region_sharpness",
    "apply_face_prior",
    "get_face_coverage_ratio",
    "visualize_face_detection",

    # EXIF 메타데이터
    "extract_exif_data",
    "compute_exif_adjustment_factors",
    "apply_exif_adjustment",
    "get_blur_risk_assessment",

    # Deep Learning NR-IQA
    "NNQuality",
    "get_model",
    "unload_model",
    "predict_quality",
    "fuse_scores",
    "nn_is_available",
    "get_device_info",

    # Google Drive 통합
    "GDriveUploader",
    "gdrive_is_available",
    "setup_credentials",
    "get_credentials_instructions",
]

# =====================================================================
# 패키지 상태 확인 함수
# =====================================================================

def check_installation() -> dict:
    """
    패키지의 설치 상태를 확인합니다.

    Returns:
        각 모듈의 사용 가능 여부를 담은 딕셔너리
    """
    status = {
        "core": _CORE_AVAILABLE,
        "io_utils": _IO_AVAILABLE,
        "helpers": _HELPERS_AVAILABLE,
        "pipeline": _PIPELINE_AVAILABLE,
        "auto_sort": _AUTO_SORT_AVAILABLE,
        "detection": _DETECTION_AVAILABLE,
        "exif": _EXIF_AVAILABLE,
        "nn_iqa": _NN_IQA_AVAILABLE,
        "gdrive": _GDRIVE_AVAILABLE,
    }
    
    # 선택적 의존성 체크
    try:
        import pillow_heif
        status["heif_support"] = True
    except ImportError:
        status["heif_support"] = False
    
    try:
        import rawpy
        status["raw_support"] = True
    except ImportError:
        status["raw_support"] = False
    
    try:
        import send2trash
        status["trash_support"] = True
    except ImportError:
        status["trash_support"] = False
    
    try:
        import torch
        status["pytorch"] = True
    except ImportError:
        status["pytorch"] = False
    
    return status


def print_status():
    """패키지 상태를 사람이 읽기 쉬운 형식으로 출력합니다."""
    status = check_installation()
    
    print("=" * 50)
    print("Unified Sort Package Status")
    print("=" * 50)
    
    print("\n[Core Modules]")
    for module in ["core", "io_utils", "helpers", "pipeline", "auto_sort", "detection", "exif", "nn_iqa", "gdrive"]:
        symbol = "✓" if status.get(module, False) else "✗"
        print(f"  {symbol} {module}")

    print("\n[Optional Features]")
    features = {
        "heif_support": "HEIC/HEIF images (iPhone photos)",
        "raw_support": "RAW image processing",
        "trash_support": "Safe delete (send to trash)",
        "pytorch": "Deep learning (PyTorch)",
    }
    
    for key, description in features.items():
        symbol = "✓" if status.get(key, False) else "✗"
        print(f"  {symbol} {description}")
    
    print("\n" + "=" * 50)

    # 권장 사항
    missing = [k for k, v in status.items() if not v and k in features]
    if missing or not status.get("gdrive", False):
        print("\nTo enable all features, install:")
        if "heif_support" in missing:
            print("  pip install pillow-heif")
        if "raw_support" in missing:
            print("  pip install rawpy imageio")
        if "trash_support" in missing:
            print("  pip install send2trash")
        if "pytorch" in missing:
            print("  pip install torch torchvision")
        if not status.get("gdrive", False):
            print("  pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")


# =====================================================================
# 편의 함수
# =====================================================================

def get_version() -> str:
    """패키지 버전을 반환합니다."""
    return __version__


def list_available_functions() -> List[str]:
    """사용 가능한 모든 공개 함수 목록을 반환합니다."""
    return [name for name in __all__ if not name.startswith("_")]


# =====================================================================
# 패키지 초기화 시 경고 표시
# =====================================================================

if not _CORE_AVAILABLE:
    print("WARNING: Core module is not available. Basic functionality will not work.")

if not _PIPELINE_AVAILABLE:
    print("INFO: Pipeline module is not available. Hybrid analysis will not work.")

# 개발 모드 체크 (선택적)
try:
    import sys
    if hasattr(sys, 'ps1'):  # 대화형 모드
        print(f"unified_sort v{__version__} loaded successfully")
        print("Run us.print_status() to check installation")
except Exception:
    pass