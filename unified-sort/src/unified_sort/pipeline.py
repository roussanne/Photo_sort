# -*- coding: utf-8 -*-
"""
Hybrid pipeline: 신호 기반 + EXIF + 얼굴 가중치 + ROI-free + 딥러닝 NR-IQA 결합
- analyze_one_full_hybrid(path, params)
- batch_analyze_full_hybrid(paths, params, ...)

개선사항:
1. 예외 처리를 구체적으로 개선 (Exception -> ImportError)
2. 딥러닝 모델 에러 처리 강화
3. 스레드 안전한 싱글톤 구현
4. 타입 검증 및 기본값 처리 개선
"""

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import cv2

# =====================================================================
# 내부 모듈 임포트 (구체적인 예외 처리)
# =====================================================================

# 핵심 분석 함수들
try:
    from .core import compute_scores_advanced
except (ImportError, ModuleNotFoundError):
    compute_scores_advanced = None

try:
    from .core import classify_object_agnostic
except (ImportError, ModuleNotFoundError, AttributeError):
    # AttributeError: core 모듈은 있지만 함수가 없는 경우
    classify_object_agnostic = None

# 얼굴 검출 기능
try:
    from .detection import apply_face_prior
except (ImportError, ModuleNotFoundError):
    apply_face_prior = None

# EXIF 메타데이터 처리
try:
    from .exif_adjust import apply_exif_adjustment
except (ImportError, ModuleNotFoundError):
    apply_exif_adjustment = None

# 딥러닝 품질 평가
try:
    from .nn_iqa import NNQuality
except (ImportError, ModuleNotFoundError):
    NNQuality = None

# 신호와 딥러닝 융합
try:
    from .hybrid import fuse_signal_and_dl
except (ImportError, ModuleNotFoundError):
    fuse_signal_and_dl = None

# 이미지 I/O (순환 임포트 방지를 위해 직접 임포트)
try:
    from .io_utils import imread_any
except (ImportError, ModuleNotFoundError):
    imread_any = None


# =====================================================================
# 폴백 함수들
# =====================================================================

def _imread_any_fallback(path: str) -> Optional[np.ndarray]:
    """
    imread_any가 없을 때 사용하는 기본 이미지 로더입니다.
    OpenCV로 BGR 형식의 numpy 배열을 반환하며, 실패 시 None을 반환합니다.
    """
    try:
        data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return data
    except Exception:
        # 파일 읽기 실패, 손상된 이미지 등 모든 경우를 안전하게 처리
        return None


def _get_img(path: str) -> Optional[np.ndarray]:
    """
    이미지를 로드하는 통합 인터페이스입니다.
    우선 imread_any를 시도하고, 없으면 폴백 함수를 사용합니다.
    """
    if callable(imread_any):
        try:
            result = imread_any(path)
            # imread_any가 예상치 못한 값을 반환할 수 있으므로 검증
            if result is not None and isinstance(result, np.ndarray):
                return result
        except Exception:
            # imread_any 실행 중 에러 발생 시 폴백으로 진행
            pass
    
    # imread_any가 없거나 실패한 경우 폴백 사용
    return _imread_any_fallback(path)


# =====================================================================
# 딥러닝 모델 관리 (스레드 안전한 싱글톤)
# =====================================================================

# 모델 인스턴스와 접근을 보호하는 락
_DL_SINGLETON: Optional[Any] = None
_DL_LOCK = threading.Lock()


def _get_dl_model(params: dict) -> Optional[Any]:
    """
    딥러닝 모델을 스레드 안전하게 로드하는 싱글톤 함수입니다.
    
    첫 호출 시 모델을 로드하고, 이후 호출에서는 캐시된 인스턴스를 반환합니다.
    여러 스레드가 동시에 호출해도 안전하도록 락을 사용합니다.
    
    Args:
        params: 모델 로딩에 필요한 파라미터 딕셔너리
        
    Returns:
        로드된 모델 인스턴스, 또는 로드 실패 시 None
    """
    global _DL_SINGLETON
    
    # 이미 로드된 경우 락 없이 빠르게 반환 (최적화)
    if _DL_SINGLETON is not None:
        return _DL_SINGLETON
    
    # NNQuality 클래스가 없으면 딥러닝을 사용할 수 없음
    if NNQuality is None:
        return None
    
    # 실제 로딩은 락으로 보호 (여러 스레드가 동시에 로드하는 것 방지)
    with _DL_LOCK:
        # 락을 획득한 후 다시 확인 (다른 스레드가 이미 로드했을 수 있음)
        if _DL_SINGLETON is not None:
            return _DL_SINGLETON
        
        try:
            weights_path = params.get("dl_weights", None)
            _DL_SINGLETON = NNQuality.from_pretrained(weights_path)
        except Exception as e:
            # 모델 로딩 실패는 치명적이지 않으므로 로그만 남기고 계속 진행
            # 실제 프로덕션에서는 logging 모듈 사용 권장
            print(f"Warning: Failed to load DL model: {e}")
            _DL_SINGLETON = None
    
    return _DL_SINGLETON


def unload_dl_model() -> None:
    """
    로드된 딥러닝 모델을 메모리에서 해제합니다.
    
    메모리가 부족하거나 모델을 더 이상 사용하지 않을 때 명시적으로 호출할 수 있습니다.
    다음 _get_dl_model 호출 시 모델이 다시 로드됩니다.
    """
    global _DL_SINGLETON
    with _DL_LOCK:
        if _DL_SINGLETON is not None:
            # PyTorch 모델인 경우 명시적으로 GPU 메모리 해제
            try:
                if hasattr(_DL_SINGLETON, 'model'):
                    import torch
                    if hasattr(_DL_SINGLETON.model, 'cpu'):
                        _DL_SINGLETON.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception:
                pass
            
            _DL_SINGLETON = None


# =====================================================================
# 점수 검증 및 정규화
# =====================================================================

def _ensure_scores_dict(scores: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    점수 딕셔너리를 안전하게 검증하고 정규화합니다.
    
    compute_scores_advanced가 예상치 못한 형식을 반환하거나,
    필수 키가 누락된 경우에도 안전하게 동작하도록 기본값을 제공합니다.
    
    Args:
        scores: 원본 점수 딕셔너리 (None일 수 있음)
        
    Returns:
        검증된 점수 딕셔너리 (항상 세 가지 키를 포함)
    """
    # 기본값: 모호한 경우를 나타내는 균등 분포
    base = {
        "sharp_score": 0.5,
        "defocus_score": 0.25,
        "motion_score": 0.25
    }
    
    # 입력이 딕셔너리가 아니면 기본값 반환
    if not isinstance(scores, dict):
        return base
    
    # 기본값으로 시작하여 유효한 값만 덮어쓰기
    result = dict(base)
    
    for key in ("sharp_score", "defocus_score", "motion_score"):
        value = scores.get(key)
        
        # 값이 존재하고 숫자로 변환 가능한 경우만 사용
        if value is not None:
            try:
                numeric_value = float(value)
                # NaN이나 무한대는 제외
                if not (np.isnan(numeric_value) or np.isinf(numeric_value)):
                    result[key] = numeric_value
            except (TypeError, ValueError):
                # 변환 실패 시 기본값 유지
                pass
    
    return result


def _clamp01(value: Any) -> float:
    """
    값을 0과 1 사이로 제한합니다.
    
    딥러닝 모델이나 계산 과정에서 범위를 벗어나는 값이 나올 수 있으므로,
    안전하게 클리핑합니다.
    
    Args:
        value: 클램핑할 값 (숫자로 변환 가능해야 함)
        
    Returns:
        0.0에서 1.0 사이의 float 값
    """
    try:
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return 0.5  # 비정상 값은 중립으로
        return max(0.0, min(1.0, numeric))
    except (TypeError, ValueError):
        # 변환 불가능한 값은 중립으로
        return 0.5


# =====================================================================
# 하이브리드 파이프라인 핵심 함수
# =====================================================================

def analyze_one_full_hybrid(path: str, params: dict) -> Dict[str, Any]:
    """
    단일 이미지에 대해 하이브리드 품질 분석을 수행합니다.
    
    이 함수는 다섯 단계의 분석 파이프라인을 실행합니다:
    1. 신호 기반 분석 (전통적 영상처리)
    2. EXIF 메타데이터 보정
    3. 얼굴 검출 및 가중치 적용
    4. ROI-free 보정 (피사체 무관)
    5. 딥러닝 NR-IQA 융합
    
    각 단계는 선택적이며, 해당 모듈이 없거나 파라미터가 비활성화되어 있으면
    스킵됩니다. 이를 통해 부분적인 기능만 있어도 동작할 수 있습니다.
    
    Args:
        path: 분석할 이미지 파일 경로
        params: 분석 파라미터 딕셔너리
            - long_side (int): 리사이즈 크기 (기본 1024)
            - tiles (int): 타일 개수 (기본 4)
            - exif_correction (bool): EXIF 보정 사용 여부
            - face_prior_enabled (bool): 얼굴 가중치 사용 여부
            - face_prior_alpha (float): 얼굴 가중치 강도 (0~1)
            - roi_free (bool): ROI-free 보정 사용 여부
            - enable_dl_hybrid (bool): 딥러닝 융합 사용 여부
            - dl_weight (float): 딥러닝 가중치 (0~1)
            - dl_motion_bias (float): 모션 바이어스 (-0.5~0.5)
            - dl_weights (str|None): 모델 가중치 파일 경로
    
    Returns:
        점수 딕셔너리 (최소한 sharp_score, defocus_score, motion_score 포함)
        실패 시 빈 딕셔너리 반환
    """
    # 이미지 로드 실패 시 조기 반환
    img_bgr = _get_img(path)
    if img_bgr is None:
        return {}
    
    # 그레이스케일 변환 (대부분의 분석 함수가 필요로 함)
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        # 이미지가 이미 그레이스케일이거나 변환 불가능한 경우
        return {}
    
    # ----------------------------------------------------------------
    # 1단계: 신호 기반 분석
    # ----------------------------------------------------------------
    tiles = int(params.get("tiles", 4))
    
    if callable(compute_scores_advanced):
        try:
            sig_scores = compute_scores_advanced(gray, tiles=tiles, params=params)
        except Exception as e:
            print(f"Warning: Signal-based analysis failed for {path}: {e}")
            sig_scores = None
    else:
        sig_scores = None
    
    # 점수를 안전하게 정규화
    scores = _ensure_scores_dict(sig_scores)
    
    # ----------------------------------------------------------------
    # 2단계: EXIF 메타데이터 보정
    # ----------------------------------------------------------------
    if params.get("exif_correction", True) and callable(apply_exif_adjustment):
        try:
            adjusted_scores = apply_exif_adjustment(path, scores)
            # 함수가 None을 반환할 수 있으므로 검증
            if isinstance(adjusted_scores, dict):
                scores = _ensure_scores_dict(adjusted_scores)
        except Exception as e:
            print(f"Warning: EXIF adjustment failed for {path}: {e}")
            # 실패해도 기존 점수 유지
    
    # ----------------------------------------------------------------
    # 3단계: 얼굴 검출 및 가중치 적용
    # ----------------------------------------------------------------
    if params.get("face_prior_enabled", True) and callable(apply_face_prior):
        try:
            alpha = _clamp01(params.get("face_prior_alpha", 0.6))
            face_scores = apply_face_prior(
                img_bgr, 
                gray, 
                scores, 
                enabled=True, 
                user_alpha=alpha
            )
            if isinstance(face_scores, dict):
                scores = _ensure_scores_dict(face_scores)
        except Exception as e:
            print(f"Warning: Face prior failed for {path}: {e}")
    
    # ----------------------------------------------------------------
    # 4단계: ROI-free 보정 (피사체 무관)
    # ----------------------------------------------------------------
    if params.get("roi_free", False) and callable(classify_object_agnostic):
        try:
            agnostic_scores = classify_object_agnostic(gray, base_scores=scores)
            if isinstance(agnostic_scores, dict):
                scores = _ensure_scores_dict(agnostic_scores)
        except Exception as e:
            print(f"Warning: ROI-free classification failed for {path}: {e}")
    
    # ----------------------------------------------------------------
    # 5단계: 딥러닝 NR-IQA 융합
    # ----------------------------------------------------------------
    if params.get("enable_dl_hybrid", True):
        dl_model = _get_dl_model(params)
        
        if dl_model is not None and callable(fuse_signal_and_dl):
            try:
                # 딥러닝 품질 점수 추론
                raw_quality = dl_model.infer_quality(img_bgr)
                
                # 안전하게 0~1 범위로 변환
                dl_quality = _clamp01(raw_quality)
                
                # 가중치와 바이어스 파라미터 추출 및 검증
                dl_weight = _clamp01(params.get("dl_weight", 0.5))
                motion_bias = float(params.get("dl_motion_bias", 0.0))
                # 바이어스는 -0.5에서 0.5 사이로 제한
                motion_bias = max(-0.5, min(0.5, motion_bias))
                
                # 신호 기반 점수와 딥러닝 점수 융합
                fused = fuse_signal_and_dl(
                    scores, 
                    dl_quality, 
                    dl_weight=dl_weight, 
                    motion_bias=motion_bias
                )
                
                if isinstance(fused, dict):
                    scores = _ensure_scores_dict(fused)
                
                # 딥러닝 점수를 메타데이터로 저장
                scores["dl_quality01"] = dl_quality
                
            except (TypeError, ValueError) as e:
                print(f"Warning: DL inference type error for {path}: {e}")
            except Exception as e:
                print(f"Warning: DL hybrid failed for {path}: {e}")
    
    # 메타데이터 추가
    scores["path"] = str(path)
    
    return scores


def _analyze_hybrid_single(args: tuple) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    단일 이미지를 하이브리드 방식으로 분석하는 헬퍼 함수 (multiprocessing용).

    ProcessPoolExecutor와 함께 사용하기 위한 top-level 함수입니다.

    Args:
        args: (path, params) 튜플

    Returns:
        (path, score_dict) 튜플, 실패 시 (path, None)
    """
    path, params = args

    try:
        score_dict = analyze_one_full_hybrid(path, params=params)
        # 유효한 결과만 반환
        if score_dict and isinstance(score_dict, dict):
            return (path, score_dict)
        else:
            return (path, None)
    except Exception as e:
        print(f"Warning: Failed to analyze {path}: {e}")
        return (path, None)


def batch_analyze_full_hybrid(
    paths: List[str],
    params: dict,
    max_workers: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    여러 이미지를 하이브리드 파이프라인으로 배치 분석합니다.

    max_workers > 1일 경우 ProcessPoolExecutor를 사용하여 병렬 처리합니다.

    Args:
        paths: 분석할 이미지 파일 경로 리스트
        params: 분석 파라미터 딕셔너리 (analyze_one_full_hybrid 참조)
        max_workers: 병렬 처리 워커 수 (1=순차, >1=병렬)

    Returns:
        경로를 키로 하는 점수 딕셔너리의 딕셔너리
        {path: {sharp_score: ..., defocus_score: ..., ...}}
    """
    results: Dict[str, Dict[str, Any]] = {}

    # 병렬 처리 또는 순차 처리 선택
    if max_workers > 1 and len(paths) > 1:
        # 병렬 처리 (CPU 집약적 작업이므로 ProcessPoolExecutor 사용)
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # 작업 인자 준비
            tasks = [(path, params) for path in paths]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 작업 제출
                future_to_path = {
                    executor.submit(_analyze_hybrid_single, task): task[0]
                    for task in tasks
                }

                # 결과 수집
                for future in as_completed(future_to_path):
                    try:
                        path, score_dict = future.result()
                        if score_dict is not None:
                            results[path] = score_dict
                    except Exception as e:
                        path = future_to_path[future]
                        print(f"Warning: Failed to get result for {path}: {e}")

        except (ImportError, OSError) as e:
            # ProcessPoolExecutor 사용 불가 시 순차 처리로 폴백
            print(f"Warning: Parallel processing failed, falling back to sequential: {e}")
            max_workers = 1

    # 순차 처리 (max_workers=1 또는 폴백)
    if max_workers == 1:
        for path in paths:
            try:
                score_dict = analyze_one_full_hybrid(path, params=params)
                # 유효한 결과만 저장
                if score_dict and isinstance(score_dict, dict):
                    results[path] = score_dict
            except Exception as e:
                print(f"Warning: Failed to analyze {path}: {e}")
                # 개별 이미지 실패가 전체 배치를 멈추지 않도록 계속 진행

    return results