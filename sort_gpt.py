# -*- coding: utf-8 -*-
"""
Powered py GPT
sort.py
모션블러 / 아웃포커스 / 선명 3-클래스 자동 점수 + 수동 라벨링 도구 (Streamlit)

[무엇을 하는가?]
- 한 폴더(선택 시 하위 폴더까지)의 이미지를 스캔해서,
  각 사진에 대해 "선명(Sharp) / 아웃포커스(Defocus) / 모션블러(Motion)" 3가지 점수를 계산합니다.
- 점수는 여러 영상처리 지표(엣지/주파수/방향성 등)를 정규화하여 가중합으로 만든 것입니다.
- 미리보기 썸네일과 자동 예측 라벨을 보여주고, 드롭다운으로 수동 교정하여 CSV로 저장할 수 있습니다.
- 라벨된 이미지를 train/{sharp, defocus, motion} 구조로 복사/이동해,
  바로 딥러닝 학습에 쓸 수 있는 데이터셋으로 내보낼 수 있습니다.

[실행 방법]
    pip install streamlit numpy pandas opencv-python Pillow
    # (HEIC 지원하려면)
    pip install pillow-heif

    streamlit run sort.py
"""

import os
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

# =============== (선택) HEIC 지원 자동 감지 ===============
try:
    import pillow_heif
    USE_HEIC = True
except Exception:
    USE_HEIC = False


# =============== 이미지 로딩 유틸 ===============
def imread_any(path: str):
    """
    경로 문자열을 받아 이미지를 BGR(cv2) 배열로 읽어온다.
    - Windows 한글/공백 경로도 안전하게 처리하기 위해 imdecode(tofile) 사용
    - iPhone HEIC/HEIF는 pillow-heif가 설치되어 있을 때만 지원
    """
    p = str(path)
    ext = p.lower().split(".")[-1]
    if USE_HEIC and ext in ("heic", "heif"):
        heif = pillow_heif.read_heif(p)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data


# =============== 저수준 특징량 함수들 ===============
def variance_of_laplacian(gray: np.ndarray) -> float:
    """Laplacian 분산: 엣지/세부(고주파)가 많을수록 값↑ → 일반적으로 '선명도'와 정비례."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad(gray: np.ndarray) -> float:
    """Tenengrad(기울기 에너지): Sobel 기울기 크기 평균. 엣지 강하면 값↑ → 초점 맞을수록 ↑."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))

def highfreq_ratio(gray: np.ndarray, cutoff: float = 0.1) -> float:
    """
    고주파 에너지 비율: 푸리에 변환(FFT)에서 중심(저주파) 밖의 고주파 에너지 비율.
    - 초점이 잘 맞은 사진은 세부 텍스처(고주파)가 많아 비율↑
    - 너무 어두운 사진이나 노이즈가 많으면 과대평가될 수 있음 → 타일링/다른 지표와 함께 사용
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = int(min(h, w) * cutoff)  # 중앙 원(저주파) 반경
    mask = (X-cx)**2 + (Y-cy)**2 > r*r
    total = np.sum(np.abs(fshift))
    high = np.sum(np.abs(fshift[mask]))
    return float(high / (total + 1e-8))

def radial_spectrum_slope(gray: np.ndarray, cutoff: float = 0.6) -> float:
    """
    Radial Spectrum Slope(라디얼 스펙트럼 기울기, 로그 스케일):
    - FFT 크기 스펙트럼을 반지름 방향으로 평균내어 1D 프로파일을 만들고,
      고주파 쪽 절반 구간의 기울기를 선형근사.
    - 아웃포커스(Defocus)일수록 고주파가 급격히 줄어 '더 음수(가파른 하강)'가 됨.
    - 값이 0에 가까울수록 고주파 유지 → 선명에 가깝고, 더 작은(음수 큰) 값일수록 아웃포커스 경향.
    """
    img = gray.astype(np.float32)
    wy = np.hanning(img.shape[0])[:, None]
    wx = np.hanning(img.shape[1])[None, :]
    win = wy * wx
    f = np.fft.fftshift(np.fft.fft2(img * win))
    mag = np.abs(f)

    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    r = R.astype(np.int32)
    rmax = int(min(h, w) * 0.5 * cutoff)

    bins = []
    for rad in range(1, rmax):
        mask = (r == rad)
        if np.any(mask):
            bins.append(np.mean(mag[mask]))
    bins = np.array(bins, dtype=np.float32)
    if len(bins) < 8:
        return 0.0

    x = np.arange(len(bins), dtype=np.float32)
    y = np.log(bins + 1e-8)
    start = len(x)//2
    Xmat = np.vstack([x[start:], np.ones_like(x[start:])]).T
    slope, _ = np.linalg.lstsq(Xmat, y[start:], rcond=None)[0]
    return float(slope)  # 더 음수일수록 defocus 쪽

def anisotropy_index(gray: np.ndarray) -> float:
    """
    방향성 지표(Anisotropy):
    - 방향별 기울기(그라디언트) 에너지 분포의 불균형 정도(표준편차).
    - 모션블러는 특정 각도에 기울기 에너지가 몰려 값↑, 아웃포커스는 대체로 등방성이라 값↓.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)  # 0~2π
    nbins = 18
    hist, _ = np.histogram(ang, bins=nbins, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return float(np.std(hist))  # ↑면 '특정 방향으로 길게 번짐(모션)'의 가능성↑

def structure_tensor_ratio(gray: np.ndarray) -> float:
    """
    구조텐서 비율(λ1-λ2)/(λ1+λ2)의 평균:
    - 영상의 국소적인 '길쭉함(타원성)'을 요약. 모션블러(한 방향으로 길게 퍼짐)일수록 ↑.
    - 텍스처가 복잡한 장면에서도 어느 정도 방향성을 잡아줄 수 있어 모션 지표를 보완.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), 1.0)
    Jyy = cv2.GaussianBlur(gy*gy, (0,0), 1.0)
    Jxy = cv2.GaussianBlur(gx*gy, (0,0), 1.0)
    l1 = 0.5*((Jxx+Jyy) + np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    l2 = 0.5*((Jxx+Jyy) - np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    num = (l1 - l2)
    den = (l1 + l2 + 1e-8)
    ratio = np.mean(num/den)
    return float(ratio)

def edge_spread_width(gray: np.ndarray, sample_edges: int = 150) -> float:
    """
    엣지 확산 폭(10%→90% 상승 거리)의 중앙값:
    - 초점이 맞을수록 엣지가 날카로워 '폭'이 작고,
      아웃포커스일수록 엣지가 퍼져 '폭'이 커진다.
    - 노이즈/질감이 강한 부분에서 오검을 줄이기 위해 다수 엣지 샘플의 중앙값 사용.
    """
    edges = cv2.Canny(gray, 80, 160)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return 0.0
    idx = np.random.choice(len(xs), size=min(sample_edges, len(xs)), replace=False)
    widths = []
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        r = 9
        y0, y1 = max(0, y-r), min(gray.shape[0], y+r+1)
        x0, x1 = max(0, x-r), min(gray.shape[1], x+r+1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 5:
            continue
        hline = np.mean(patch, axis=0)
        vline = np.mean(patch, axis=1)
        for arr in (hline, vline):
            a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            p10 = np.argmax(a >= 0.1)
            p90 = np.argmax(a >= 0.9)
            if p90 > p10:
                widths.append(p90 - p10)
    return float(np.median(widths) if widths else 0.0)


# =============== 타일링 집계 ===============
def tile_features(gray: np.ndarray, tiles: int = 4) -> dict:
    """
    이미지를 NxN 타일로 나눠 각 타일에서 위 지표들을 계산하고,
    '취약한 타일'의 퍼센타일로 보수적으로 집계한다.
    - vol/ten/hfr/slope는 하위 20% (가장 흐린 지역에 민감)
    - esw/aniso/strat는 상위 80% (가장 넓게 퍼지고/방향성이 큰 지역에 민감)
    """
    H, W = gray.shape
    hs, ws = H//tiles, W//tiles
    vols, tens, hfrs, esws, anisos, slopes, strats = [], [], [], [], [], [], []
    for i in range(tiles):
        for j in range(tiles):
            crop = gray[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            if crop.size < 20:
                continue
            vols.append(variance_of_laplacian(crop))
            tens.append(tenengrad(crop))
            hfrs.append(highfreq_ratio(crop))
            esws.append(edge_spread_width(crop, sample_edges=60))
            anisos.append(anisotropy_index(crop))
            slopes.append(radial_spectrum_slope(crop))
            strats.append(structure_tensor_ratio(crop))

    def p(a, q): return float(np.percentile(a, q)) if a else 0.0

    return {
        "vol_p20":   p(vols,  20),
        "ten_p20":   p(tens,  20),
        "hfr_p20":   p(hfrs,  20),
        "esw_p80":   p(esws,  80),
        "aniso_p80": p(anisos, 80),
        "slope_p20": p(slopes, 20),
        "strat_p80": p(strats, 80),
    }


# =============== 정규화 & 3-클래스 스코어링 ===============
def norm_box(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """값 x를 [lo,hi] 구간에서 [0,1]로 선형 매핑(클립). invert면 1-x."""
    x = float(x)
    v = (x - lo) / (hi - lo + 1e-8)
    v = min(max(v, 0.0), 1.0)
    return 1.0 - v if invert else v

def compute_scores(gray: np.ndarray, tiles: int, params: dict) -> dict:
    """
    최종 3-클래스 점수 계산:
      Sharp  = vol_n + ten_n + hfr_n + (esw_n) + (slope_n)
      Defocus= (1-esw_n) + (1-vol_n) + (1-slope_n) + (1-aniso_n)
      Motion = aniso_n + strat_n + (1-vol_n)
    각 항목 앞의 가중치는 사이드바 슬라이더로 조절.
    """
    H, W = gray.shape
    long_side = params["long_side"]
    if max(H, W) > long_side:
        s = long_side / max(H, W)
        gray = cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    F = tile_features(gray, tiles=tiles)

    # 경험적 범위(필요시 코드에서 조절 가능)
    vol_n   = norm_box(F["vol_p20"],   50,   600,  invert=False)  # 선명↑
    ten_n   = norm_box(F["ten_p20"],   1.0,  12.0, invert=False)  # 선명↑
    hfr_n   = norm_box(F["hfr_p20"],   0.02, 0.35, invert=False)  # 선명↑
    esw_n   = norm_box(F["esw_p80"],   1.0,  6.0,  invert=True)   # 선명↑
    slope_n = norm_box(F["slope_p20"], -6.0, -0.5, invert=True)   # 선명↑
    aniso_n = norm_box(F["aniso_p80"], 0.0,  0.12, invert=False)  # 모션↑
    strat_n = norm_box(F["strat_p80"], 0.02, 0.45, invert=False)  # 모션↑

    sharp_score = (
        params["w_sharp_vol"] * vol_n +
        params["w_sharp_ten"] * ten_n +
        params["w_sharp_hfr"] * hfr_n +
        params["w_sharp_esw"] * esw_n +
        params["w_sharp_slope"] * slope_n
    )
    defocus_score = (
        params["w_def_esw"] * (1 - esw_n) +
        params["w_def_vol"] * (1 - vol_n) +
        params["w_def_slope"] * (1 - slope_n) +
        params["w_def_aniso"] * (1 - aniso_n)
    )
    motion_score = (
        params["w_mot_aniso"] * aniso_n +
        params["w_mot_strat"] * strat_n +
        params["w_mot_volinv"] * (1 - vol_n)
    )
    return {
        "features": F,
        "sharp_score": float(sharp_score),
        "defocus_score": float(defocus_score),
        "motion_score": float(motion_score),
    }


# =============== 캐시: 목록/썸네일/점수 ===============
@st.cache_data(show_spinner=False)
def list_images(root: str, recursive: bool = False):
    patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
                "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
    if USE_HEIC:
        patterns += ["*.heic","*.heif","*.HEIC","*.HEIF"]
    paths = []
    if recursive:
        for pat in patterns:
            paths += list(Path(root).rglob(pat))
    else:
        for pat in patterns:
            paths += list(Path(root).glob(pat))
    paths = [p for p in paths if p.is_file()]
    return [str(p) for p in sorted(set(paths))]

@st.cache_data(show_spinner=False)
def load_thumbnail(path: str, max_side: int = 384):
    img = imread_any(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@st.cache_data(show_spinner=False)
def compute_scores_cached(path: str, tiles: int, params: dict):
    img = imread_any(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return compute_scores(gray, tiles=tiles, params=params)


# =============== Streamlit UI ===============
st.set_page_config(page_title="3-클래스 분류/라벨링", layout="wide")
st.title("📷 모션블러 / 아웃포커스 / 선명 — 3-클래스 분류 + 수동 라벨링 툴")

with st.sidebar:
    st.header("폴더 & 스캔")
    root = st.text_input(
        "이미지 폴더 경로",
        value=str(Path.home() / "Pictures"),
        help=(
            "분석할 이미지들이 들어 있는 폴더 경로를 입력합니다.\n\n"
            "💡 팁: 처음에는 100~300장 정도의 작은 폴더로 실험해보세요.\n"
            "데이터 성격(야경/인물/망원 등)에 따라 가중치 최적값이 달라지고,\n"
            "최소 스코어(threshold)도 조정이 필요할 수 있습니다."
        )
    )
    recursive = st.checkbox(
        "하위 폴더까지 포함",
        value=False,
        help=(
            "체크하면 선택한 폴더의 모든 하위 폴더까지 재귀적으로 스캔합니다.\n\n"
            "주의: 이미지가 매우 많다면 초기 스캔 시간이 길어질 수 있습니다."
        )
    )
    long_side = st.slider(
        "분석용 리사이즈(긴 변)",
        min_value=640, max_value=2048, value=1024, step=64,
        help=(
            "점수 계산 전에 이미지를 이 길이에 맞춰 축소합니다(긴 변 기준).\n\n"
            "왜 필요? 서로 다른 해상도에서 지표 값이 크게 요동치는 것을 완화합니다.\n"
            "값이 너무 작으면 디테일이 사라져 과도한 '흐림' 판정이 나올 수 있고,\n"
            "너무 크면 처리 속도가 느려집니다. 보통 896~1280 사이가 무난합니다."
        )
    )

    st.header("타일 / 가중치")
    tiles = st.slider(
        "타일 수 (NxN)",
        min_value=2, max_value=6, value=4, step=1,
        help=(
            "이미지를 NxN 개의 타일로 나눠 지역별로 특징을 계측합니다.\n\n"
            "이유: 사진의 일부분만 흐린 경우(구석/배경)도 잡아내기 위함입니다.\n"
            "값이 클수록(=타일 많을수록) 국소 흐림 탐지가 잘 되지만 처리 시간이 늘어납니다.\n"
            "권장: 3~5. 야경/망원은 4~6으로 올려보세요."
        )
    )

    st.caption("— Sharp(선명) 가중치 —")
    w_sharp_vol   = st.slider(
        "VoL",
        0.0, 1.0, 0.30, 0.01,
        help=(
            "Laplacian Variance(선명도 지표) 가중치.\n"
            "• 의미: 모서리/고주파가 많을수록 값이 커지며, 일반적으로 선명함과 상관이 큽니다.\n"
            "• 높일수록 '엣지가 많은 사진'을 더 선명으로 판단합니다.\n"
            "• 노이즈가 많은 고감도 사진에서는 값이 과대평가될 수 있으므로\n"
            "  HighFreqRatio/ESW/기울기와 균형을 맞추세요."
        )
    )
    w_sharp_ten   = st.slider(
        "Tenengrad",
        0.0, 1.0, 0.25, 0.01,
        help=(
            "Sobel 기울기 에너지 기반 선명 가중치.\n"
            "• 의미: 엣지 강도가 클수록 선명한 사진으로 판단합니다.\n"
            "• 잔무늬/텍스처가 많은 배경(잔디, 천)에서 과대평가되는 경향이 있어\n"
            "  다른 지표와 함께 쓰는 것이 좋습니다."
        )
    )
    w_sharp_hfr   = st.slider(
        "HighFreqRatio",
        0.0, 1.0, 0.20, 0.01,
        help=(
            "고주파 성분 비율 가중치.\n"
            "• 의미: 세부 텍스처가 많으면 값↑. 초점이 맞은 사진에서 일반적으로 높습니다.\n"
            "• 야경 노이즈/강한 ISO 노이즈는 고주파로 잡혀 과대평가될 수 있습니다.\n"
            "  ESW/라디얼 기울기와 함께 보정하세요."
        )
    )
    w_sharp_esw   = st.slider(
        "EdgeSpread(역)",
        0.0, 1.0, 0.15, 0.01,
        help=(
            "엣지 폭의 '역수' 가중치(엣지가 얇을수록 선명으로 가중).\n"
            "• 의미: 초점이 맞으면 엣지가 얇고, 아웃포커스면 두꺼워집니다.\n"
            "• 조언: 인물 배경보케가 많은 사진에서 ESW가 커질 수 있으므로,\n"
            "  이 값을 과도하게 올리면 배경 아웃포커스를 '흐림'으로 과잉 판정할 수 있습니다."
        )
    )
    w_sharp_slope = st.slider(
        "RadialSlope(역)",
        0.0, 1.0, 0.10, 0.01,
        help=(
            "라디얼 스펙트럼 기울기의 '역수' 가중치(덜 음수일수록 선명 가중).\n"
            "• 의미: 아웃포커스일수록 고주파 급감 → 기울기 더 음수.\n"
            "• 이 값을 높이면 '고주파 보존' 특성이 강한 사진이 선명으로 가중됩니다."
        )
    )

    st.caption("— Defocus(아웃포커스) 가중치 —")
    w_def_esw   = st.slider(
        "EdgeSpread",
        0.0, 1.0, 0.40, 0.01,
        help=(
            "엣지 폭 가중치(클수록 아웃포커스 가중↑).\n"
            "• 의미: 엣지가 두꺼울수록(퍼질수록) Defocus로 분류하는 경향을 강화합니다.\n"
            "• 권장: 배경이 자연스럽게 흐릿한 사진(보케)도 많은 데이터라면\n"
            "  이 가중치를 과도하게 높이지 않도록 주의하세요."
        )
    )
    w_def_vol   = st.slider(
        "VoL(역)",
        0.0, 1.0, 0.25, 0.01,
        help=(
            "Laplacian Variance 역가중치(낮을수록 Defocus 가중↑).\n"
            "• 의미: 고주파가 부족하면 Defocus로 분류될 가능성을 높입니다.\n"
            "• VoL은 노이즈/피사체 종류에 민감할 수 있으므로 다른 지표와 함께 쓰세요."
        )
    )
    w_def_slope = st.slider(
        "RadialSlope(역)",
        0.0, 1.0, 0.25, 0.01,
        help=(
            "라디얼 스펙트럼 기울기 역가중치(더 음수일수록 Defocus 가중↑).\n"
            "• 의미: 고주파 감소가 빠른(블러) 이미지를 Defocus로 강하게 잡습니다."
        )
    )
    w_def_aniso = st.slider(
        "Anisotropy(역)",
        0.0, 1.0, 0.10, 0.01,
        help=(
            "방향성 지표의 '역' 가중치(등방성일수록 Defocus 가중↑).\n"
            "• 의미: 모션블러는 특정 방향으로 길게 퍼지지만, 아웃포커스는 전반적으로 퍼집니다.\n"
            "  따라서 '방향성 낮음(등방성)'은 Defocus의 단서가 됩니다."
        )
    )

    st.caption("— Motion(모션블러) 가중치 —")
    w_mot_aniso   = st.slider(
        "Anisotropy",
        0.0, 1.0, 0.60, 0.01,
        help=(
            "방향성(이방성) 가중치(특정 방향으로 흐릴수록 값↑).\n"
            "• 의미: 모션블러의 핵심 단서로, 한 방향으로 길게 늘어진 흔적을 포착합니다.\n"
            "• 야간 차량 궤적/패닝샷 등에서는 높게 설정하는 것이 유리합니다."
        )
    )
    w_mot_strat   = st.slider(
        "StructureTensor",
        0.0, 1.0, 0.30, 0.01,
        help=(
            "구조텐서 기반 타원성 가중치(길쭉함↑ → 모션 가중↑).\n"
            "• 의미: 단순 방향성(hist 편차) 외에도 지역적 길쭉함을 포착해 모션블러를 보완합니다."
        )
    )
    w_mot_volinv  = st.slider(
        "VoL(역)",
        0.0, 1.0, 0.10, 0.01,
        help=(
            "VoL의 역가중치(흐릴수록 모션 가중↑).\n"
            "• 의미: 모션이든 아웃이든 흐리면 VoL이 낮아지지만,\n"
            "  다른 모션 지표(Anisotropy/구조텐서)와 조합해 모션 쪽으로 더 기울입니다."
        )
    )

    st.header("분류/필터 옵션")
    min_sharp = st.slider(
        "선명 최소 스코어",
        0.0, 1.0, 0.35, 0.01,
        help=(
            "자동 예측이 '선명'이라도 이 값보다 낮으면 다른 클래스(아웃/모션)로 재할당합니다.\n\n"
            "언제 올리나? 선명으로 분류된 사진 중 흐릿한 컷이 섞일 때.\n"
            "언제 내리나? 선명 컷을 놓치는 경우(재현율을 높이고 싶을 때)."
        )
    )
    min_def   = st.slider(
        "아웃포커스 최소 스코어",
        0.0, 1.0, 0.35, 0.01,
        help=(
            "자동 예측이 '아웃포커스'라도 이 값보다 낮으면 다른 클래스(선명/모션)로 재할당합니다.\n\n"
            "언제 올리나? Defocus를 더욱 엄격하게 잡고 싶을 때(강한 흐림만 남김).\n"
            "언제 내리나? 약한 아웃포커스도 잡고 싶을 때(재현율↑)."
        )
    )
    min_mot   = st.slider(
        "모션 최소 스코어",
        0.0, 1.0, 0.35, 0.01,
        help=(
            "자동 예측이 '모션'이라도 이 값보다 낮으면 다른 클래스(선명/아웃)로 재할당합니다.\n\n"
            "언제 올리나? 모션블러만 엄격히 추리려 할 때.\n"
            "언제 내리나? 약한 모션도 포함하고 싶을 때."
        )
    )
    show_pred = st.selectbox(
        "미리보기 필터",
        ["모두", "선명", "아웃포커스", "모션블러"],
        index=0,
        help=(
            "썸네일 미리보기를 특정 예측 클래스로 필터링합니다.\n"
            "예: '모션블러'만 선택하면 모션으로 예측(또는 보정 후 유지)된 항목만 표시."
        )
    )

# 파라미터 묶기
params = dict(
    long_side=long_side,
    w_sharp_vol=w_sharp_vol, w_sharp_ten=w_sharp_ten, w_sharp_hfr=w_sharp_hfr,
    w_sharp_esw=w_sharp_esw, w_sharp_slope=w_sharp_slope,
    w_def_esw=w_def_esw, w_def_vol=w_def_vol, w_def_slope=w_def_slope, w_def_aniso=w_def_aniso,
    w_mot_aniso=w_mot_aniso, w_mot_strat=w_mot_strat, w_mot_volinv=w_mot_volinv,
)

# 이미지 목록/페이지네이션
paths = list_images(root, recursive=recursive)
st.write(f"**총 이미지**: {len(paths)}")

cols = st.columns(2)
with cols[0]:
    per_page = st.selectbox(
        "페이지당 썸네일 수", [12, 24, 48], index=1,
        help="한 페이지에 표시할 썸네일 개수입니다. 느리면 개수를 줄여보세요."
    )
with cols[1]:
    page = st.number_input(
        "페이지(1부터)",
        min_value=1, value=1, step=1,
        help="몇 번째 페이지를 볼지 지정합니다."
    )
start = (page-1)*per_page
end = min(start+per_page, len(paths))
page_paths = paths[start:end]

# 세션 상태(라벨/점수 저장소)
if "labels" not in st.session_state:
    st.session_state["labels"] = {}
if "scores" not in st.session_state:
    st.session_state["scores"] = {}

# 썸네일 그리드 (4열)
grid_cols = st.columns(4)
for i, p in enumerate(page_paths):
    col = grid_cols[i % 4]
    with col:
        thumb = load_thumbnail(p, max_side=384)
        if thumb is not None:
            st.image(thumb, use_column_width=True)

        S = compute_scores_cached(p, tiles=tiles, params=params)
        if S is None:
            st.caption("읽기 실패")
            continue

        sharp_s = S["sharp_score"]
        def_s   = S["defocus_score"]
        mot_s   = S["motion_score"]

        # 1차: argmax
        preds = [("sharp", sharp_s), ("defocus", def_s), ("motion", mot_s)]
        pred = max(preds, key=lambda x: x[1])[0]
        # 2차: 최소 스코어 기준으로 재할당(약한 신호 억제)
        if pred == "sharp" and sharp_s < min_sharp:
            pred = "defocus" if def_s >= max(min_def, mot_s) else "motion"
        if pred == "defocus" and def_s < min_def:
            pred = "sharp" if sharp_s >= max(min_sharp, mot_s) else "motion"
        if pred == "motion" and mot_s < min_mot:
            pred = "sharp" if sharp_s >= max(min_sharp, def_s) else "defocus"

        # 미리보기 필터 적용
        if show_pred != "모두":
            need = {"선명":"sharp", "아웃포커스":"defocus", "모션블러":"motion"}[show_pred]
            if pred != need:
                continue

        st.session_state["scores"][p] = dict(sharp=sharp_s, defocus=def_s, motion=mot_s)

        # 수동 라벨 드롭다운(자동 예측값을 기본으로 제안)
        current = st.session_state["labels"].get(p, pred)
        new_label = st.selectbox(
            label=f"{Path(p).name}\nS:{sharp_s:.2f} D:{def_s:.2f} M:{mot_s:.2f}",
            options=["sharp","defocus","motion"],
            index=["sharp","defocus","motion"].index(current),
            key=f"sel_{p}",
            help=(
                "자동 예측값을 기본으로 표시합니다. 사람이 실제로 보기엔 다른 경우가 많습니다.\n"
                "여기서 원하는 클래스로 수동으로 교정해 주세요.\n\n"
                "📌 라벨링 팁:\n"
                "• 모션블러: 특정 방향으로 줄무늬/늘어짐이 보임(피사체 전체가 한쪽으로 흔들린 느낌)\n"
                "• 아웃포커스: 방향성 없이 전반적으로 퍼짐(보케/둥근 하이라이트가 배경에 많을 수 있음)\n"
                "• 선명: 엣지가 날카롭고, 미세 질감이 잘 보임"
            )
        )
        st.session_state["labels"][p] = new_label

st.divider()

# 일괄 적용/저장/내보내기
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    bulk_label = st.selectbox(
        "페이지 전체 라벨",
        ["(선택 없음)","sharp","defocus","motion"],
        help=(
            "현재 페이지에 표시된 항목 모두에 같은 라벨을 한 번에 적용합니다.\n"
            "비슷한 사진들이 한 페이지에 몰려 있을 때 빠르게 라벨링할 수 있습니다."
        )
    )
with c2:
    if st.button("이 페이지에 일괄 적용"):
        for p in page_paths:
            if p in st.session_state["scores"] and bulk_label != "(선택 없음)":
                st.session_state["labels"][p] = bulk_label
        st.success("이 페이지 항목에 일괄 적용 완료")

with c3:
    if st.button("CSV 저장 (labels.csv)"):
        rows = []
        for p in paths:
            if p in st.session_state["scores"]:
                lab = st.session_state["labels"].get(p, "sharp")
                sc  = st.session_state["scores"][p]
                rows.append({
                    "path": p, "label": lab,
                    "sharp_score": round(sc["sharp"],4),
                    "defocus_score": round(sc["defocus"],4),
                    "motion_score": round(sc["motion"],4),
                })
        df = pd.DataFrame(rows)
        out_csv = Path(root) / "labels.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        st.success(f"저장: {out_csv}\n"
                   f"→ 이 파일은 모델 학습 전 데이터 점검/통계/샘플링에도 유용합니다.")

with c4:
    move_or_copy = st.selectbox(
        "내보내기 방식", ["copy","move"],
        help=(
            "라벨된 이미지를 학습용 폴더로 보낼 때 '복사(copy)' 또는 '이동(move)' 중 선택합니다.\n"
            "• copy: 원본은 그대로 두고, 학습 폴더에 복사본을 만듭니다(안전).\n"
            "• move: 원본을 학습 폴더로 옮깁니다(정리 용이하나 되돌리기 어려움)."
        )
    )
with c5:
    if st.button("학습셋으로 내보내기 (train/)"):
        out_root = Path(root) / "train"
        (out_root / "sharp").mkdir(parents=True, exist_ok=True)
        (out_root / "defocus").mkdir(parents=True, exist_ok=True)
        (out_root / "motion").mkdir(parents=True, exist_ok=True)

        n_done = 0
        for p in paths:
            if p not in st.session_state["labels"]:
                continue
            lab = st.session_state["labels"][p]
            dst = out_root / lab / Path(p).name
            try:
                if move_or_copy == "copy":
                    shutil.copy2(p, dst)
                else:
                    shutil.move(p, dst)
                n_done += 1
            except Exception as e:
                st.warning(f"처리 실패: {p} -> {dst} ({e})")

        st.success(
            f"학습셋 내보내기 완료 ({move_or_copy}): {n_done}개 → {out_root}\n\n"
            "📦 폴더 구조 예: \n"
            "train/\n"
            " ├─ sharp/\n"
            " ├─ defocus/\n"
            " └─ motion/\n"
            "→ PyTorch torchvision.datasets.ImageFolder, tf.data 등에서 바로 사용 가능합니다."
        )

# 요약
st.subheader("라벨 요약")
if st.session_state["labels"]:
    ser = pd.Series(st.session_state["labels"]).value_counts()
    st.write(ser.to_frame("count"))
else:
    st.write("아직 라벨이 없습니다. 썸네일 드롭다운으로 라벨을 지정해 보세요.")

# 끝!
