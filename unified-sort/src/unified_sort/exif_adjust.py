"""
EXIF metadata extraction and adjustment module.

카메라 설정 기반 점수 조정:
- ISO: 높은 ISO → 노이즈 증가, 모션블러 가능성 증가
- 셔터 속도: 느린 셔터 → 모션블러 위험
- 조리개: 넓은 조리개 → 얕은 피사계 심도 (디포커스 가능성)
- 초점 거리: 망원 렌즈 → 디포커스 민감도 증가

개선사항:
1. PIL과 piexif 사용으로 폭넓은 EXIF 지원
2. 타입 힌트 및 입력 검증
3. 구체적 예외 처리
4. 정규화된 조정 계수 계산
"""

from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np


# EXIF 태그 상수 (표준 Exif 2.3)
EXIF_TAG_ISO = 34855  # ISOSpeedRatings
EXIF_TAG_SHUTTER_SPEED = 33434  # ExposureTime
EXIF_TAG_APERTURE = 33437  # FNumber
EXIF_TAG_FOCAL_LENGTH = 37386  # FocalLength
EXIF_TAG_EXPOSURE_MODE = 34850  # ExposureMode
EXIF_TAG_FLASH = 37385  # Flash


def _safe_rational_to_float(value: Any) -> Optional[float]:
    """
    EXIF rational 값을 float으로 안전하게 변환합니다.

    Args:
        value: EXIF rational 값 (tuple, int, float 등)

    Returns:
        float 값, 변환 실패 시 None
    """
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            # Rational (numerator, denominator)
            numerator, denominator = value
            if denominator == 0:
                return None
            return float(numerator) / float(denominator)
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_exif_data(image_path: str) -> Dict[str, Any]:
    """
    이미지 파일에서 EXIF 메타데이터를 추출합니다.

    PIL(Pillow)을 사용하여 JPEG, TIFF 등의 EXIF 데이터를 읽습니다.

    Args:
        image_path: 이미지 파일 경로

    Returns:
        EXIF 데이터 딕셔너리 {
            'iso': int | None,
            'shutter_speed': float | None (초 단위),
            'aperture': float | None (f-number),
            'focal_length': float | None (mm),
            'exposure_mode': int | None,
            'flash_fired': bool | None
        }
        추출 실패 시 모든 값이 None인 딕셔너리 반환
    """
    # 기본값 (모든 None)
    exif_data = {
        'iso': None,
        'shutter_speed': None,
        'aperture': None,
        'focal_length': None,
        'exposure_mode': None,
        'flash_fired': None
    }

    # 파일 존재 확인
    if not Path(image_path).exists():
        return exif_data

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        # 이미지 열기
        img = Image.open(image_path)

        # EXIF 데이터 가져오기
        exifdata = img.getexif()

        if not exifdata:
            return exif_data

        # ISO
        if EXIF_TAG_ISO in exifdata:
            iso_value = exifdata[EXIF_TAG_ISO]
            if isinstance(iso_value, (int, list, tuple)):
                # ISO는 리스트일 수 있음 (ISO 100, 200 등)
                if isinstance(iso_value, (list, tuple)) and len(iso_value) > 0:
                    exif_data['iso'] = int(iso_value[0])
                else:
                    exif_data['iso'] = int(iso_value)

        # 셔터 속도 (ExposureTime)
        if EXIF_TAG_SHUTTER_SPEED in exifdata:
            shutter = _safe_rational_to_float(exifdata[EXIF_TAG_SHUTTER_SPEED])
            if shutter is not None:
                exif_data['shutter_speed'] = shutter

        # 조리개 (F-Number)
        if EXIF_TAG_APERTURE in exifdata:
            aperture = _safe_rational_to_float(exifdata[EXIF_TAG_APERTURE])
            if aperture is not None:
                exif_data['aperture'] = aperture

        # 초점 거리
        if EXIF_TAG_FOCAL_LENGTH in exifdata:
            focal_length = _safe_rational_to_float(exifdata[EXIF_TAG_FOCAL_LENGTH])
            if focal_length is not None:
                exif_data['focal_length'] = focal_length

        # 노출 모드
        if EXIF_TAG_EXPOSURE_MODE in exifdata:
            exif_data['exposure_mode'] = int(exifdata[EXIF_TAG_EXPOSURE_MODE])

        # 플래시 발광 여부
        if EXIF_TAG_FLASH in exifdata:
            flash_value = int(exifdata[EXIF_TAG_FLASH])
            # 최하위 비트가 1이면 플래시 발광
            exif_data['flash_fired'] = bool(flash_value & 0x1)

    except (ImportError, OSError, KeyError, ValueError, AttributeError) as e:
        # PIL 없음, 파일 손상, EXIF 없음 등
        pass

    return exif_data


def compute_exif_adjustment_factors(exif_data: Dict[str, Any]) -> Dict[str, float]:
    """
    EXIF 데이터로부터 점수 조정 계수를 계산합니다.

    각 카메라 설정이 블러 타입에 미치는 영향을 분석하여
    조정 계수를 반환합니다.

    Args:
        exif_data: extract_exif_data()로 추출한 EXIF 딕셔너리

    Returns:
        조정 계수 딕셔너리 {
            'motion_bias': float (-0.3 ~ +0.3),
            'defocus_bias': float (-0.3 ~ +0.3),
            'sharp_bias': float (-0.3 ~ +0.3),
            'confidence_penalty': float (0.0 ~ 0.5)
        }
    """
    factors = {
        'motion_bias': 0.0,
        'defocus_bias': 0.0,
        'sharp_bias': 0.0,
        'confidence_penalty': 0.0
    }

    # === ISO 분석 ===
    # 높은 ISO → 손떨림 보정 부족 or 어두운 환경 → 모션블러 가능성
    if exif_data.get('iso') is not None:
        iso = exif_data['iso']
        if iso > 3200:
            factors['motion_bias'] += 0.15
            factors['confidence_penalty'] += 0.1
        elif iso > 1600:
            factors['motion_bias'] += 0.08
        elif iso > 800:
            factors['motion_bias'] += 0.03

    # === 셔터 속도 분석 ===
    # 느린 셔터 → 모션블러 위험 증가
    if exif_data.get('shutter_speed') is not None:
        shutter = exif_data['shutter_speed']

        # 1/30초보다 느리면 손떨림 위험
        if shutter > 1.0 / 30.0:
            # 1/15초 → 매우 높은 위험
            if shutter > 1.0 / 15.0:
                factors['motion_bias'] += 0.25
                factors['confidence_penalty'] += 0.15
            # 1/30초 → 중간 위험
            else:
                factors['motion_bias'] += 0.15
                factors['confidence_penalty'] += 0.08

        # 매우 빠른 셔터 → 모션블러 가능성 낮음
        elif shutter < 1.0 / 500.0:
            factors['motion_bias'] -= 0.1
            factors['sharp_bias'] += 0.05

    # === 조리개 분석 ===
    # 넓은 조리개 (낮은 f-number) → 얕은 피사계 심도 → 디포커스 가능성
    if exif_data.get('aperture') is not None:
        aperture = exif_data['aperture']

        # f/1.4 ~ f/2.8 → 매우 얕은 피사계 심도
        if aperture < 2.8:
            factors['defocus_bias'] += 0.15
        # f/2.8 ~ f/4.0 → 얕은 피사계 심도
        elif aperture < 4.0:
            factors['defocus_bias'] += 0.08
        # f/8.0 이상 → 깊은 피사계 심도 (디포커스 가능성 낮음)
        elif aperture >= 8.0:
            factors['defocus_bias'] -= 0.1
            factors['sharp_bias'] += 0.05

    # === 초점 거리 분석 ===
    # 망원 렌즈 → 디포커스 민감도 증가 & 손떨림 영향 증가
    if exif_data.get('focal_length') is not None:
        focal_length = exif_data['focal_length']

        # 200mm 이상 → 망원
        if focal_length > 200:
            factors['defocus_bias'] += 0.1
            factors['motion_bias'] += 0.12
            factors['confidence_penalty'] += 0.08
        # 100-200mm → 중망원
        elif focal_length > 100:
            factors['defocus_bias'] += 0.05
            factors['motion_bias'] += 0.06
        # 24mm 이하 → 광각 (손떨림 영향 적음)
        elif focal_length < 24:
            factors['motion_bias'] -= 0.08

    # === 플래시 분석 ===
    # 플래시 사용 → 모션블러 가능성 감소 (짧은 발광 시간)
    if exif_data.get('flash_fired') is True:
        factors['motion_bias'] -= 0.15
        factors['sharp_bias'] += 0.08

    # 계수 클리핑 (과도한 조정 방지)
    for key in ['motion_bias', 'defocus_bias', 'sharp_bias']:
        factors[key] = max(-0.3, min(0.3, factors[key]))

    factors['confidence_penalty'] = max(0.0, min(0.5, factors['confidence_penalty']))

    return factors


def apply_exif_adjustment(
    scores: Dict[str, float],
    exif_data: Dict[str, Any],
    adjustment_strength: float = 0.5
) -> Dict[str, float]:
    """
    EXIF 메타데이터 기반으로 점수를 조정합니다.

    카메라 설정 정보를 활용하여 블러 분류 점수를 미세 조정합니다.

    Args:
        scores: 원본 점수 딕셔너리 {sharp_score, defocus_score, motion_score}
        exif_data: extract_exif_data()로 추출한 EXIF 데이터
        adjustment_strength: 조정 강도 (0.0 ~ 1.0, 기본값 0.5)

    Returns:
        조정된 점수 딕셔너리
    """
    # 입력 검증
    if not isinstance(scores, dict) or not isinstance(exif_data, dict):
        return scores

    required_keys = {"sharp_score", "defocus_score", "motion_score"}
    if not required_keys.issubset(scores.keys()):
        return scores

    # EXIF 데이터가 없으면 원본 반환
    if all(v is None for v in exif_data.values()):
        return scores

    # 조정 계수 계산
    factors = compute_exif_adjustment_factors(exif_data)

    # 조정 강도 클리핑
    strength = max(0.0, min(1.0, adjustment_strength))

    # 점수 복사
    adjusted_scores = scores.copy()

    # 바이어스 적용
    adjusted_scores['sharp_score'] += factors['sharp_bias'] * strength
    adjusted_scores['defocus_score'] += factors['defocus_bias'] * strength
    adjusted_scores['motion_score'] += factors['motion_bias'] * strength

    # 음수 클리핑
    for key in adjusted_scores:
        adjusted_scores[key] = max(0.0, adjusted_scores[key])

    # 정규화 (합 = 1)
    total = sum(adjusted_scores.values())
    if total > 0:
        for key in adjusted_scores:
            adjusted_scores[key] /= total

    # 신뢰도 패널티 적용 (선택적)
    # 반환값에 메타데이터 추가
    if 'metadata' not in adjusted_scores:
        adjusted_scores['metadata'] = {}

    adjusted_scores['metadata']['exif_confidence_penalty'] = factors['confidence_penalty']
    adjusted_scores['metadata']['exif_applied'] = True
    adjusted_scores['metadata']['exif_summary'] = _generate_exif_summary(exif_data)

    return adjusted_scores


def _generate_exif_summary(exif_data: Dict[str, Any]) -> str:
    """
    EXIF 데이터의 요약 문자열을 생성합니다.

    Args:
        exif_data: EXIF 딕셔너리

    Returns:
        사람이 읽기 쉬운 요약 문자열
    """
    parts = []

    if exif_data.get('iso') is not None:
        parts.append(f"ISO {exif_data['iso']}")

    if exif_data.get('shutter_speed') is not None:
        shutter = exif_data['shutter_speed']
        if shutter < 1.0:
            parts.append(f"1/{int(1.0/shutter)}s")
        else:
            parts.append(f"{shutter:.1f}s")

    if exif_data.get('aperture') is not None:
        parts.append(f"f/{exif_data['aperture']:.1f}")

    if exif_data.get('focal_length') is not None:
        parts.append(f"{int(exif_data['focal_length'])}mm")

    if exif_data.get('flash_fired'):
        parts.append("Flash")

    return ", ".join(parts) if parts else "No EXIF"


def get_blur_risk_assessment(exif_data: Dict[str, Any]) -> Dict[str, str]:
    """
    EXIF 데이터를 기반으로 블러 위험도를 평가합니다.

    Args:
        exif_data: EXIF 딕셔너리

    Returns:
        위험도 평가 딕셔너리 {
            'motion_risk': str ('Low', 'Medium', 'High'),
            'defocus_risk': str ('Low', 'Medium', 'High'),
            'overall_assessment': str
        }
    """
    factors = compute_exif_adjustment_factors(exif_data)

    # 모션블러 위험도
    if factors['motion_bias'] > 0.15:
        motion_risk = 'High'
    elif factors['motion_bias'] > 0.05:
        motion_risk = 'Medium'
    else:
        motion_risk = 'Low'

    # 디포커스 위험도
    if factors['defocus_bias'] > 0.12:
        defocus_risk = 'High'
    elif factors['defocus_bias'] > 0.05:
        defocus_risk = 'Medium'
    else:
        defocus_risk = 'Low'

    # 전체 평가
    if motion_risk == 'High' or defocus_risk == 'High':
        overall = 'High blur risk detected from camera settings'
    elif motion_risk == 'Medium' or defocus_risk == 'Medium':
        overall = 'Moderate blur risk from camera settings'
    else:
        overall = 'Low blur risk (good camera settings)'

    return {
        'motion_risk': motion_risk,
        'defocus_risk': defocus_risk,
        'overall_assessment': overall
    }
