"""
Core analysis functions for image quality assessment.

개선사항:
1. 타입 힌트 추가로 명확성 향상
2. 에러 처리 강화
3. 입력 검증 추가
4. 문서화 개선
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2


def list_images(root: str, recursive: bool = False) -> List[str]:
    """
    지정된 폴더에서 이미지 파일을 찾아 경로 리스트를 반환합니다.
    
    지원하는 형식: JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP (대소문자 무관)
    
    Args:
        root: 검색할 루트 폴더 경로
        recursive: True면 하위 폴더까지 재귀적으로 검색
    
    Returns:
        발견된 이미지 파일의 절대 경로 리스트 (정렬됨)
        폴더가 존재하지 않으면 빈 리스트 반환
    """
    root_path = Path(root)
    
    # 폴더가 존재하지 않거나 유효하지 않은 경로인 경우
    if not root_path.exists():
        return []
    
    if not root_path.is_dir():
        return []
    
    # 지원하는 이미지 확장자 패턴 (대소문자 모두)
    patterns = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP"
    ]
    
    paths = []
    
    try:
        if recursive:
            # 재귀적 검색 (모든 하위 폴더 포함)
            for pattern in patterns:
                paths.extend(root_path.rglob(pattern))
        else:
            # 현재 폴더만 검색
            for pattern in patterns:
                paths.extend(root_path.glob(pattern))
    except (OSError, PermissionError) as e:
        # 접근 권한이 없거나 파일 시스템 오류
        print(f"Warning: Error scanning directory {root}: {e}")
        return []
    
    # Path 객체를 문자열로 변환하고, 파일만 필터링한 후 정렬
    valid_paths = [str(p) for p in paths if p.is_file()]
    
    # 중복 제거 후 정렬 (대소문자가 다른 같은 파일 처리)
    return sorted(set(valid_paths))


def load_thumbnail(path: str, max_side: int = 384) -> Optional[np.ndarray]:
    """
    이미지를 썸네일 크기로 로드합니다.
    
    원본 이미지의 비율을 유지하면서 긴 변이 max_side를 넘지 않도록
    리사이즈합니다. BGR 형식의 numpy 배열로 반환합니다.
    
    Args:
        path: 이미지 파일 경로
        max_side: 썸네일의 최대 변 길이 (픽셀)
    
    Returns:
        BGR 형식의 numpy 배열 (uint8), 실패 시 None
    """
    from .io_utils import imread_any
    
    if not isinstance(max_side, int) or max_side <= 0:
        max_side = 384  # 기본값으로 복구
    
    try:
        img = imread_any(path)
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return None
    
    if img is None:
        return None
    
    # 이미지가 유효한 numpy 배열인지 확인
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
    
    try:
        h, w = img.shape[:2]
        
        # 이미 충분히 작으면 리사이즈 불필요
        if max(h, w) <= max_side:
            return img
        
        # 비율을 유지하면서 축소
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 최소 크기 보장 (1픽셀 이상)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        img_resized = cv2.resize(
            img, 
            (new_w, new_h), 
            interpolation=cv2.INTER_AREA  # 축소 시 가장 좋은 품질
        )
        
        return img_resized
        
    except Exception as e:
        print(f"Warning: Failed to resize thumbnail for {path}: {e}")
        return None


def compute_scores_advanced(
    gray: np.ndarray, 
    tiles: int, 
    params: dict
) -> Dict[str, float]:
    """
    고급 영상처리 기법으로 이미지 품질 점수를 계산합니다.
    
    이 함수는 실제 구현의 플레이스홀더입니다. 프로덕션에서는
    실제 7가지 특징 추출 로직으로 대체해야 합니다:
    - Variance of Laplacian (VoL)
    - Tenengrad
    - High Frequency Ratio
    - Edge Spread Width
    - Radial Spectrum Slope
    - Anisotropy Index
    - Structure Tensor Ratio
    
    Args:
        gray: 그레이스케일 이미지 (numpy 배열)
        tiles: 타일 분할 개수 (NxN)
        params: 분석 파라미터 딕셔너리
    
    Returns:
        세 가지 점수를 포함하는 딕셔너리
        {sharp_score: float, defocus_score: float, motion_score: float}
    """
    # 입력 검증
    if not isinstance(gray, np.ndarray):
        return {
            "sharp_score": 0.5,
            "defocus_score": 0.25,
            "motion_score": 0.25
        }
    
    if gray.size == 0:
        return {
            "sharp_score": 0.5,
            "defocus_score": 0.25,
            "motion_score": 0.25
        }
    
    # TODO: 실제 구현으로 대체
    # 현재는 고정값 반환 (UI 테스트용)
    return {
        "sharp_score": 0.5,
        "defocus_score": 0.3,
        "motion_score": 0.2,
    }


def batch_analyze(
    paths: List[str], 
    mode: str = "simple", 
    tiles: int = 4, 
    params: Optional[dict] = None, 
    max_workers: int = 1
) -> Dict[str, dict]:
    """
    여러 이미지를 배치로 분석합니다.
    
    간단 모드(simple)는 라플라시안 기반의 빠른 분석을,
    고급 모드(advanced)는 더 정교한 다중 특징 분석을 수행합니다.
    
    Args:
        paths: 분석할 이미지 경로 리스트
        mode: "simple" 또는 "advanced"
        tiles: 타일 개수 (advanced 모드에서만 사용)
        params: 추가 분석 파라미터
        max_workers: 병렬 처리 워커 수 (현재 미사용)
    
    Returns:
        {경로: 점수딕셔너리} 형태의 결과
    """
    from .io_utils import imread_any
    
    if params is None:
        params = {}
    
    results = {}
    
    for path in paths:
        try:
            img = imread_any(path)
            if img is None:
                continue
            
            # 그레이스케일 변환
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                # 이미 그레이스케일이거나 변환 불가능
                continue
            
            if mode == "simple":
                # 간단 모드: 라플라시안 기반 빠른 분석
                try:
                    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                    edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                    edge_mean = float(np.mean(np.abs(edge)))
                    
                    # 0-100 스케일로 정규화
                    sharpness_score = min(100.0, lap / 5.0)
                    edge_score = min(100.0, edge_mean / 0.1)
                    
                    # 가중 평균
                    combined_score = 0.6 * sharpness_score + 0.4 * edge_score
                    
                    # 타입 판별 (간단한 임계값 기반)
                    if combined_score > 60:
                        blur_type = "선명 ✅"
                        quality = "좋음"
                    else:
                        # 방향성 체크 (간단 버전)
                        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                        mag = np.sqrt(gx*gx + gy*gy) + 1e-8
                        ang = (np.arctan2(gy, gx) + np.pi)
                        hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
                        direction_std = float(np.std(hist / (hist.sum() + 1e-8)))
                        
                        if direction_std > 0.08:
                            blur_type = "모션블러 📸"
                            quality = "흐림 (움직임)"
                        else:
                            blur_type = "아웃포커스 🌫️"
                            quality = "흐림 (초점)"
                    
                    results[path] = {
                        "score": round(combined_score, 1),
                        "type": blur_type,
                        "quality": quality,
                        "laplacian": round(lap, 2),
                        "edge": round(edge_mean, 2),
                        "direction": 0.0,
                    }
                    
                except Exception as e:
                    print(f"Warning: Simple analysis failed for {path}: {e}")
                    continue
                    
            else:
                # 고급 모드: 다중 특징 기반 분석
                try:
                    results[path] = compute_scores_advanced(
                        gray, 
                        tiles=tiles, 
                        params=params
                    )
                except Exception as e:
                    print(f"Warning: Advanced analysis failed for {path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            continue
    
    return results