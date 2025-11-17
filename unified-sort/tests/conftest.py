"""
Pytest configuration and fixtures.

테스트용 공통 설정 및 픽스처:
- 샘플 이미지 생성 (다양한 품질/형식)
- 임시 디렉토리 관리
- 모의(mock) 객체 설정
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
from typing import Generator, Dict, List


@pytest.fixture(scope="session")
def temp_test_dir() -> Generator[Path, None, None]:
    """
    세션 전체에서 사용할 임시 테스트 디렉토리를 생성합니다.

    Yields:
        임시 디렉토리 경로
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="unified_sort_test_"))
    yield temp_dir
    # 테스트 종료 후 정리
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_sharp_image(temp_test_dir: Path) -> Path:
    """
    선명한 테스트 이미지를 생성합니다.

    Args:
        temp_test_dir: 임시 디렉토리

    Returns:
        생성된 이미지 파일 경로
    """
    # 고주파 패턴 생성 (선명한 이미지)
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 체커보드 패턴
    for i in range(0, 256, 16):
        for j in range(0, 256, 16):
            if (i // 16 + j // 16) % 2 == 0:
                img[i:i+16, j:j+16] = [255, 255, 255]

    # 저장
    filepath = temp_test_dir / "sharp_image.jpg"
    cv2.imwrite(str(filepath), img)

    return filepath


@pytest.fixture
def sample_defocus_image(temp_test_dir: Path) -> Path:
    """
    디포커스 블러 이미지를 생성합니다.

    Args:
        temp_test_dir: 임시 디렉토리

    Returns:
        생성된 이미지 파일 경로
    """
    # 선명한 패턴 생성
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, (255, 255, 255), -1)
    cv2.circle(img, (128, 128), 40, (0, 0, 0), -1)

    # 가우시안 블러 적용 (디포커스 시뮬레이션)
    img = cv2.GaussianBlur(img, (15, 15), 5.0)

    # 저장
    filepath = temp_test_dir / "defocus_image.jpg"
    cv2.imwrite(str(filepath), img)

    return filepath


@pytest.fixture
def sample_motion_blur_image(temp_test_dir: Path) -> Path:
    """
    모션 블러 이미지를 생성합니다.

    Args:
        temp_test_dir: 임시 디렉토리

    Returns:
        생성된 이미지 파일 경로
    """
    # 선명한 패턴 생성
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(10, 246, 20):
        cv2.line(img, (10, i), (246, i), (255, 255, 255), 2)

    # 모션 블러 커널 생성
    kernel_size = 15
    kernel_motion = np.zeros((kernel_size, kernel_size))
    kernel_motion[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion = kernel_motion / kernel_size

    # 모션 블러 적용
    img = cv2.filter2D(img, -1, kernel_motion)

    # 저장
    filepath = temp_test_dir / "motion_blur_image.jpg"
    cv2.imwrite(str(filepath), img)

    return filepath


@pytest.fixture
def sample_images(
    sample_sharp_image: Path,
    sample_defocus_image: Path,
    sample_motion_blur_image: Path
) -> Dict[str, Path]:
    """
    모든 샘플 이미지를 딕셔너리로 반환합니다.

    Returns:
        {label: filepath} 매핑
    """
    return {
        "sharp": sample_sharp_image,
        "defocus": sample_defocus_image,
        "motion": sample_motion_blur_image
    }


@pytest.fixture
def sample_image_batch(temp_test_dir: Path) -> List[Path]:
    """
    다양한 품질의 이미지 배치를 생성합니다.

    Args:
        temp_test_dir: 임시 디렉토리

    Returns:
        이미지 파일 경로 리스트
    """
    images = []

    # 5개의 선명한 이미지
    for i in range(5):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        filepath = temp_test_dir / f"batch_sharp_{i}.jpg"
        cv2.imwrite(str(filepath), img)
        images.append(filepath)

    # 3개의 흐린 이미지
    for i in range(3):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (11, 11), 3.0)
        filepath = temp_test_dir / f"batch_blur_{i}.jpg"
        cv2.imwrite(str(filepath), img)
        images.append(filepath)

    return images


@pytest.fixture
def mock_exif_data() -> Dict[str, any]:
    """
    모의 EXIF 데이터를 반환합니다.

    Returns:
        EXIF 데이터 딕셔너리
    """
    return {
        'iso': 400,
        'shutter_speed': 1/125,  # 1/125초
        'aperture': 2.8,
        'focal_length': 50.0,
        'exposure_mode': 0,
        'flash_fired': False
    }


@pytest.fixture
def mock_high_iso_exif() -> Dict[str, any]:
    """
    높은 ISO의 모의 EXIF 데이터를 반환합니다 (모션블러 위험).

    Returns:
        EXIF 데이터 딕셔너리
    """
    return {
        'iso': 6400,  # 높은 ISO
        'shutter_speed': 1/30,  # 느린 셔터
        'aperture': 1.4,  # 넓은 조리개
        'focal_length': 200.0,  # 망원
        'exposure_mode': 0,
        'flash_fired': False
    }


@pytest.fixture
def sample_scores() -> Dict[str, float]:
    """
    샘플 분류 점수를 반환합니다.

    Returns:
        점수 딕셔너리
    """
    return {
        "sharp_score": 0.7,
        "defocus_score": 0.2,
        "motion_score": 0.1
    }


@pytest.fixture
def sample_uncertain_scores() -> Dict[str, float]:
    """
    불확실한 분류 점수를 반환합니다 (마진이 작음).

    Returns:
        점수 딕셔너리
    """
    return {
        "sharp_score": 0.4,
        "defocus_score": 0.35,
        "motion_score": 0.25
    }


@pytest.fixture
def sample_low_quality_scores() -> Dict[str, float]:
    """
    전체적으로 낮은 품질 점수를 반환합니다.

    Returns:
        점수 딕셔너리
    """
    return {
        "sharp_score": 0.15,
        "defocus_score": 0.12,
        "motion_score": 0.08
    }


# 스킵 조건 정의
def pytest_configure(config):
    """pytest 설정을 초기화합니다."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "torch_required: marks tests requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "gdrive_required: marks tests requiring Google Drive API"
    )


# PyTorch 사용 가능 여부 확인
pytest_plugins = []

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Google Drive API 사용 가능 여부 확인
try:
    from google.oauth2.credentials import Credentials
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False


# 조건부 스킵을 위한 마커
skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)

skip_if_no_gdrive = pytest.mark.skipif(
    not GDRIVE_AVAILABLE,
    reason="Google Drive API not installed"
)
