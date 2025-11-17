"""
Face detection and weighting module for image quality assessment.

얼굴 검출 기반 가중치 적용:
- 얼굴이 있는 사진의 경우 얼굴 영역의 선명도를 우선시
- 배경 블러(bokeh)를 잘못된 디포커스로 분류하지 않도록 방지
- OpenCV Haar Cascade 기반 빠른 검출

개선사항:
1. 구체적 예외 처리 (ImportError vs OSError)
2. 타입 힌트 추가
3. 입력 검증 강화
4. 캐스케이드 파일 자동 검색
"""

from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import cv2


# 전역 캐스케이드 로더 (싱글톤 패턴)
_face_cascade: Optional[cv2.CascadeClassifier] = None


def _load_face_cascade() -> Optional[cv2.CascadeClassifier]:
    """
    얼굴 검출용 Haar Cascade 로더.

    OpenCV에 내장된 haarcascade_frontalface_default.xml을 로드합니다.
    싱글톤 패턴으로 한 번만 로드하여 성능을 향상시킵니다.

    Returns:
        CascadeClassifier 객체, 실패 시 None
    """
    global _face_cascade

    if _face_cascade is not None:
        return _face_cascade

    try:
        # OpenCV 데이터 경로에서 캐스케이드 파일 검색
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        # 파일 존재 확인
        if not Path(cascade_path).exists():
            print(f"Warning: Haar cascade file not found at {cascade_path}")
            return None

        # 로드
        cascade = cv2.CascadeClassifier(cascade_path)

        if cascade.empty():
            print("Warning: Failed to load Haar cascade (empty classifier)")
            return None

        _face_cascade = cascade
        return _face_cascade

    except (ImportError, OSError, AttributeError) as e:
        print(f"Warning: Could not load face cascade: {e}")
        return None


def detect_faces(
    gray: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30)
) -> List[Tuple[int, int, int, int]]:
    """
    이미지에서 얼굴을 검출합니다.

    Args:
        gray: 그레이스케일 이미지 (numpy 배열)
        scale_factor: 이미지 피라미드 스케일 팩터 (1.1 ~ 1.5 권장)
        min_neighbors: 얼굴로 판정하기 위한 최소 이웃 개수 (높을수록 정확하지만 놓칠 수 있음)
        min_size: 검출할 얼굴의 최소 크기 (픽셀)

    Returns:
        검출된 얼굴 영역 리스트 [(x, y, w, h), ...]
        검출 실패 시 빈 리스트 반환
    """
    # 입력 검증
    if not isinstance(gray, np.ndarray) or gray.size == 0:
        return []

    if len(gray.shape) != 2:
        # 그레이스케일이 아닌 경우 변환 시도
        try:
            if len(gray.shape) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                return []
        except Exception:
            return []

    # 캐스케이드 로드
    cascade = _load_face_cascade()
    if cascade is None:
        return []

    try:
        # 얼굴 검출
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # NumPy 배열을 리스트로 변환
        if len(faces) == 0:
            return []

        # (x, y, w, h) 튜플 리스트로 변환
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    except Exception as e:
        print(f"Warning: Face detection failed: {e}")
        return []


def compute_face_region_sharpness(
    gray: np.ndarray,
    faces: List[Tuple[int, int, int, int]]
) -> float:
    """
    얼굴 영역의 평균 선명도를 계산합니다.

    각 얼굴 영역에 대해 Laplacian 분산을 계산하고 평균을 반환합니다.

    Args:
        gray: 그레이스케일 이미지
        faces: 얼굴 영역 리스트 [(x, y, w, h), ...]

    Returns:
        얼굴 영역의 평균 선명도 점수 (Laplacian 분산)
        얼굴이 없거나 실패 시 0.0 반환
    """
    if not faces or not isinstance(gray, np.ndarray):
        return 0.0

    sharpness_scores = []

    for x, y, w, h in faces:
        try:
            # 경계 확인
            h_img, w_img = gray.shape
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            # 얼굴 영역 추출
            face_region = gray[y1:y2, x1:x2]

            if face_region.size < 100:  # 너무 작은 영역 제외
                continue

            # Laplacian 분산 계산
            laplacian = cv2.Laplacian(face_region, cv2.CV_64F)
            variance = laplacian.var()

            sharpness_scores.append(float(variance))

        except Exception as e:
            print(f"Warning: Failed to compute sharpness for face region: {e}")
            continue

    if not sharpness_scores:
        return 0.0

    return float(np.mean(sharpness_scores))


def apply_face_prior(
    scores: Dict[str, float],
    gray: np.ndarray,
    face_weight: float = 0.6
) -> Dict[str, float]:
    """
    얼굴 검출 결과를 활용하여 점수에 가중치를 적용합니다.

    얼굴이 검출되면:
    1. 얼굴 영역의 선명도를 계산
    2. 얼굴이 선명하고 배경이 흐린 경우 (의도된 bokeh) → sharp 점수 증가
    3. 얼굴도 흐린 경우 → 원본 점수 유지 또는 defocus/motion 점수 증가

    Args:
        scores: 원본 분류 점수 {sharp_score, defocus_score, motion_score}
        gray: 그레이스케일 이미지
        face_weight: 얼굴 선명도의 가중치 (0.0 ~ 1.0)

    Returns:
        조정된 점수 딕셔너리
    """
    # 입력 검증
    if not isinstance(scores, dict) or not isinstance(gray, np.ndarray):
        return scores

    required_keys = {"sharp_score", "defocus_score", "motion_score"}
    if not required_keys.issubset(scores.keys()):
        return scores

    # 얼굴 검출
    faces = detect_faces(gray)

    # 얼굴이 없으면 원본 점수 반환
    if not faces:
        return scores

    # 얼굴 영역 선명도 계산
    face_sharpness = compute_face_region_sharpness(gray, faces)

    # 전체 이미지 선명도 계산 (비교용)
    try:
        full_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        full_sharpness = 0.0

    # 얼굴이 배경보다 훨씬 선명한 경우 (의도된 bokeh 효과)
    if face_sharpness > full_sharpness * 1.5 and face_sharpness > 100:
        # Sharp 점수 증가, defocus 점수 감소
        adjusted_scores = scores.copy()

        boost_factor = min(0.3, (face_sharpness / (full_sharpness + 1e-6) - 1.0) * 0.1)

        adjusted_scores["sharp_score"] = min(1.0, scores["sharp_score"] + boost_factor)
        adjusted_scores["defocus_score"] = max(0.0, scores["defocus_score"] - boost_factor * 0.5)

        # 정규화
        total = sum(adjusted_scores.values())
        if total > 0:
            for key in adjusted_scores:
                adjusted_scores[key] /= total

        return adjusted_scores

    # 얼굴도 흐린 경우 (실제 블러)
    elif face_sharpness < 50:
        # 원본 점수 유지 (얼굴도 흐리므로 실제 블러)
        return scores

    # 중간 경우 (애매한 경우)
    else:
        # 약한 가중치 적용
        adjusted_scores = scores.copy()

        # 얼굴 선명도를 0-1 범위로 정규화
        face_sharpness_norm = min(1.0, face_sharpness / 200.0)

        # 가중 평균
        weight = face_weight * 0.5  # 약한 가중치
        adjusted_scores["sharp_score"] = (
            scores["sharp_score"] * (1 - weight) +
            face_sharpness_norm * weight
        )

        # 정규화
        total = sum(adjusted_scores.values())
        if total > 0:
            for key in adjusted_scores:
                adjusted_scores[key] /= total

        return adjusted_scores


def get_face_coverage_ratio(
    gray: np.ndarray,
    faces: List[Tuple[int, int, int, int]]
) -> float:
    """
    이미지 내 얼굴이 차지하는 영역 비율을 계산합니다.

    Args:
        gray: 그레이스케일 이미지
        faces: 얼굴 영역 리스트

    Returns:
        얼굴 영역 비율 (0.0 ~ 1.0)
    """
    if not faces or not isinstance(gray, np.ndarray) or gray.size == 0:
        return 0.0

    h_img, w_img = gray.shape
    total_area = h_img * w_img

    if total_area == 0:
        return 0.0

    face_area = 0
    for x, y, w, h in faces:
        # 이미지 경계 내로 클리핑
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        if x2 > x1 and y2 > y1:
            face_area += (x2 - x1) * (y2 - y1)

    coverage = face_area / total_area
    return float(min(1.0, coverage))


def visualize_face_detection(
    img_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    검출된 얼굴에 사각형을 그려 시각화합니다.

    Args:
        img_bgr: BGR 컬러 이미지
        faces: 얼굴 영역 리스트
        color: 사각형 색상 (B, G, R)
        thickness: 선 두께

    Returns:
        얼굴 사각형이 그려진 이미지 (원본 수정하지 않음)
    """
    if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return img_bgr

    # 원본 이미지 복사
    img_viz = img_bgr.copy()

    for x, y, w, h in faces:
        try:
            cv2.rectangle(img_viz, (x, y), (x + w, y + h), color, thickness)
        except Exception:
            continue

    return img_viz
