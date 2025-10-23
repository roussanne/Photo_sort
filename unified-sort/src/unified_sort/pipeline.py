# -*- coding: utf-8 -*-
"""
Hybrid pipeline: 신호 기반 + EXIF + 얼굴 가중치 + ROI-free + 딥러닝 NR-IQA 결합
- analyze_one_full_hybrid(path, params)
- batch_analyze_full_hybrid(paths, params, ...)
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import cv2

# 내부 모듈 (있으면 사용, 없으면 안전 폴백)
try:
    from .core import compute_scores_advanced, classify_object_agnostic  # classify_object_agnostic 없을 수 있음
except Exception:
    compute_scores_advanced = None
    classify_object_agnostic = None

try:
    from .detection import apply_face_prior  # 선택
except Exception:
    apply_face_prior = None

try:
    from .exif_adjust import apply_exif_adjustment  # 선택
except Exception:
    apply_exif_adjustment = None

try:
    from .nn_iqa import NNQuality  # 선택
except Exception:
    NNQuality = None

try:
    from .hybrid import fuse_signal_and_dl  # 선택
except Exception:
    fuse_signal_and_dl = None

# ✅ 순환 임포트 방지: 루트(__init__)가 아닌 실제 모듈에서 직접 import
try:
    from .io_utils import imread_any  # 권장
except Exception:
    imread_any = None


def _imread_any_fallback(path: str):
    """imread_any 부재 시 폴백(BGR np.ndarray)"""
    try:
        data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return data
    except Exception:
        return None


def _get_img(path: str):
    if callable(imread_any):
        try:
            return imread_any(path)
        except Exception:
            pass
    return _imread_any_fallback(path)


# 딥러닝 모델 lazy singleton
_DL_SINGLETON: Optional[object] = None

def _get_dl_model(params: dict):
    global _DL_SINGLETON
    if _DL_SINGLETON is not None:
        return _DL_SINGLETON
    if NNQuality is None:
        return None
    try:
        _DL_SINGLETON = NNQuality.from_pretrained(params.get("dl_weights", None))
    except Exception:
        _DL_SINGLETON = None
    return _DL_SINGLETON


def _ensure_scores_dict(scores: dict | None) -> Dict[str, float]:
    """
    compute_scores_advanced 결과를 방어적으로 보정.
    최소 필수 키가 없으면 기본값 대입.
    """
    base = {"sharp_score": 0.5, "defocus_score": 0.25, "motion_score": 0.25}
    if not isinstance(scores, dict):
        return base
    out = dict(base)
    for k in ("sharp_score", "defocus_score", "motion_score"):
        v = scores.get(k)
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def analyze_one_full_hybrid(path: str, params: dict) -> Dict:
    """
    1장에 대해 하이브리드 품질 점수를 계산하여 dict 반환.

    params (주요):
      - long_side (int): 신호기반 리사이즈 (core 내부에서 사용)
      - tiles (int): 타일 개수 (기본 4)
      - exif_correction (bool)
      - face_prior_enabled (bool)
      - face_prior_alpha (float)
      - roi_free (bool)
      - enable_dl_hybrid (bool)
      - dl_weight (float, 0~1)
      - dl_motion_bias (float)
      - dl_weights (str|None)
    """
    img_bgr = _get_img(path)
    if img_bgr is None:
        return {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- (1) 신호 기반 ---
    tiles = int(params.get("tiles", 4))
    if callable(compute_scores_advanced):
        try:
            sig_scores = compute_scores_advanced(gray, tiles=tiles, params=params)
        except Exception:
            sig_scores = None
    else:
        sig_scores = None

    scores = _ensure_scores_dict(sig_scores)

    # --- (2) EXIF 보정 ---
    if params.get("exif_correction", True) and callable(apply_exif_adjustment):
        try:
            scores = apply_exif_adjustment(path, scores) or scores
        except Exception:
            pass

    # --- (3) 얼굴(인물) 가중치 ---
    if params.get("face_prior_enabled", True) and callable(apply_face_prior):
        try:
            alpha = float(params.get("face_prior_alpha", 0.6))
            scores = apply_face_prior(img_bgr, gray, scores, enabled=True, user_alpha=alpha) or scores
        except Exception:
            pass

    # --- (4) ROI-free (피사체 무관 보정) ---
    if params.get("roi_free", False) and callable(classify_object_agnostic):
        try:
            scores = classify_object_agnostic(gray, base_scores=scores) or scores
        except Exception:
            pass

    # --- (5) 딥러닝 NR-IQA 융합 ---
    if params.get("enable_dl_hybrid", True):
        dl_model = _get_dl_model(params)
        if dl_model is not None and callable(fuse_signal_and_dl):
            try:
                dl_q = float(dl_model.infer_quality(img_bgr))  # 0~1
                dl_q = _clamp01(dl_q)
                dl_w = _clamp01(params.get("dl_weight", 0.5))
                mot_bias = float(params.get("dl_motion_bias", 0.0))
                fused = fuse_signal_and_dl(scores, dl_q, dl_weight=dl_w, motion_bias=mot_bias)
                if isinstance(fused, dict):
                    scores = _ensure_scores_dict(fused)
                scores["dl_quality01"] = dl_q
            except Exception:
                pass

    # 메타
    scores["path"] = str(path)
    return scores


def batch_analyze_full_hybrid(
    paths: List[str],
    params: dict,
    max_workers: int = 1
) -> Dict[str, Dict]:
    """
    여러 장을 하이브리드 파이프라인으로 분석.
    (필요 시 여기에 멀티프로세싱/스레딩 적용)
    """
    results: Dict[str, Dict] = {}
    for p in paths:
        try:
            S = analyze_one_full_hybrid(p, params=params)
            if S:
                results[p] = S
        except Exception:
            pass
    return results
