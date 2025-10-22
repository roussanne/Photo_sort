# -*- coding: utf-8 -*-
"""
Unified Image Quality Classifier
통합 이미지 품질 검사 도구

특징:
- 간단 모드 / 고급 모드 전환 가능
- 일반 사용자부터 전문가까지 모두 사용 가능
- 하나의 인터페이스로 모든 기능 제공
"""

import os
import math
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# =============== 선택적 라이브러리 임포트 ===============
try:
    import pillow_heif
    USE_HEIC = True
except Exception:
    USE_HEIC = False

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    USE_CNN = True
except Exception:
    USE_CNN = False


# =============== 이미지 로딩 유틸 ===============
def imread_any(path: str):
    """이미지를 읽어오는 함수입니다. HEIC 포맷과 일반 이미지를 모두 지원합니다."""
    p = str(path)
    ext = p.lower().split(".")[-1]
    if USE_HEIC and ext in ("heic", "heif"):
        heif = pillow_heif.read_heif(p)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data


# =============== 간단 모드용 선명도 측정 ===============
def check_sharpness_simple(gray: np.ndarray) -> dict:
    """
    간단 모드용 선명도 체크 함수입니다.
    빠른 처리를 위해 핵심 지표만 계산하며, 0에서 100까지의 점수로 변환합니다.
    """
    h, w = gray.shape
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        gray = cv2.resize(gray, (int(w*s), int(h*s)))
    
    # 라플라시안 분산으로 선명도를 측정합니다
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 소벨 필터로 엣지 강도를 측정합니다
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(gx*gx + gy*gy))
    
    # 방향성을 계산하여 모션블러를 감지합니다
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)
    hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    direction_score = np.std(hist)
    
    # 점수를 0에서 100 사이로 정규화합니다
    sharpness_score = min(100, (laplacian_var / 5.0))
    edge_score = min(100, (edge_strength / 0.1))
    combined_score = (sharpness_score * 0.6 + edge_score * 0.4)
    
    # 점수와 방향성을 기반으로 흐림 타입을 판단합니다
    if combined_score > 60:
        blur_type = "선명 ✅"
        quality = "좋음"
    elif direction_score > 0.08:
        blur_type = "모션블러 📸"
        quality = "흐림 (움직임)"
    else:
        blur_type = "아웃포커스 🌫️"
        quality = "흐림 (초점)"
    
    return {
        "score": round(combined_score, 1),
        "type": blur_type,
        "quality": quality,
        "laplacian": round(laplacian_var, 2),
        "edge": round(edge_strength, 2),
        "direction": round(direction_score, 3)
    }


# =============== 고급 모드용 저수준 특징량 함수들 ===============
def variance_of_laplacian(gray: np.ndarray) -> float:
    """라플라시안 분산을 계산합니다. 높을수록 이미지가 선명합니다."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad(gray: np.ndarray) -> float:
    """그라디언트 에너지를 계산합니다. 엣지가 강할수록 값이 높습니다."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))

def highfreq_ratio(gray: np.ndarray, cutoff: float = 0.1) -> float:
    """푸리에 변환을 사용하여 고주파 성분의 비율을 계산합니다."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = int(min(h, w) * cutoff)
    mask = (X-cx)**2 + (Y-cy)**2 > r*r
    total = np.sum(np.abs(fshift))
    high = np.sum(np.abs(fshift[mask]))
    return float(high / (total + 1e-8))

def radial_spectrum_slope(gray: np.ndarray, cutoff: float = 0.6) -> float:
    """
    라디얼 스펙트럼의 기울기를 계산합니다.
    아웃포커스일수록 고주파가 급격히 감소하여 기울기가 더 음수가 됩니다.
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
    return float(slope)

def anisotropy_index(gray: np.ndarray) -> float:
    """
    방향성 지표를 계산합니다.
    모션블러는 특정 방향으로 기울기가 몰려 있어 값이 높습니다.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)
    nbins = 18
    hist, _ = np.histogram(ang, bins=nbins, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return float(np.std(hist))

def structure_tensor_ratio(gray: np.ndarray) -> float:
    """
    구조 텐서를 사용하여 국소적 타원성을 계산합니다.
    모션블러일수록 한 방향으로 길게 늘어나 값이 높습니다.
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
    엣지의 확산 폭을 측정합니다.
    초점이 맞을수록 엣지가 좁고, 흐릴수록 넓어집니다.
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


# =============== 고급 모드용 타일링 집계 ===============
def tile_features(gray: np.ndarray, tiles: int = 4) -> dict:
    """
    이미지를 여러 타일로 나누어 각 타일의 특징을 계산합니다.
    이를 통해 이미지의 일부만 흐린 경우도 감지할 수 있습니다.
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


# =============== 정규화 & 고급 모드 스코어링 ===============
def norm_box(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """값을 0에서 1 사이로 정규화합니다."""
    x = float(x)
    v = (x - lo) / (hi - lo + 1e-8)
    v = min(max(v, 0.0), 1.0)
    return 1.0 - v if invert else v

def compute_scores_advanced(gray: np.ndarray, tiles: int, params: dict) -> dict:
    """
    고급 모드용 점수 계산 함수입니다.
    여러 특징을 조합하여 3-클래스 점수를 계산합니다.
    """
    H, W = gray.shape
    long_side = params["long_side"]
    if max(H, W) > long_side:
        s = long_side / max(H, W)
        gray = cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    F = tile_features(gray, tiles=tiles)

    # 각 특징을 0-1 사이로 정규화합니다
    vol_n   = norm_box(F["vol_p20"],   50,   600,  invert=False)
    ten_n   = norm_box(F["ten_p20"],   1.0,  12.0, invert=False)
    hfr_n   = norm_box(F["hfr_p20"],   0.02, 0.35, invert=False)
    esw_n   = norm_box(F["esw_p80"],   1.0,  6.0,  invert=True)
    slope_n = norm_box(F["slope_p20"], -6.0, -0.5, invert=True)
    aniso_n = norm_box(F["aniso_p80"], 0.0,  0.12, invert=False)
    strat_n = norm_box(F["strat_p80"], 0.02, 0.45, invert=False)

    # 가중합으로 각 클래스의 점수를 계산합니다
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
        "normalized": {
            "vol_n": vol_n, "ten_n": ten_n, "hfr_n": hfr_n,
            "esw_n": esw_n, "slope_n": slope_n, "aniso_n": aniso_n, "strat_n": strat_n
        },
        "sharp_score": float(sharp_score),
        "defocus_score": float(defocus_score),
        "motion_score": float(motion_score),
    }


# =============== CNN 모델 (실험적) ===============
class BlurClassifierCNN:
    """ResNet18 기반 분류 모델입니다. 딥러닝을 활용한 분류를 제공합니다."""
    def __init__(self):
        if not USE_CNN:
            self.model = None
            return
        
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, img_bgr: np.ndarray) -> dict:
        """이미지를 입력받아 3-클래스 확률을 반환합니다."""
        if self.model is None:
            return {"sharp": 0.33, "defocus": 0.33, "motion": 0.34}
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        return {
            "sharp": float(probs[0]),
            "defocus": float(probs[1]),
            "motion": float(probs[2])
        }


# =============== 멀티프로세싱 배치 처리 ===============
def process_single_image(args):
    """멀티프로세싱을 위한 단일 이미지 처리 함수입니다."""
    path, mode, tiles, params = args
    try:
        img = imread_any(path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if mode == "simple":
            return (path, check_sharpness_simple(gray))
        else:
            return (path, compute_scores_advanced(gray, tiles=tiles, params=params))
    except Exception as e:
        return None

def batch_process_images(paths, mode, tiles, params, max_workers=None):
    """
    여러 이미지를 병렬로 처리합니다.
    진행률을 표시하여 사용자가 처리 상황을 확인할 수 있습니다.
    """
    if max_workers is None:
        max_workers = min(cpu_count(), 8)
    
    args_list = [(p, mode, tiles, params) for p in paths]
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with Pool(processes=max_workers) as pool:
        for i, result in enumerate(pool.imap(process_single_image, args_list), 1):
            if result:
                path, scores = result
                results[path] = scores
            
            progress = i / len(paths)
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {i}/{len(paths)} ({progress*100:.1f}%)")
    
    progress_bar.empty()
    status_text.empty()
    
    return results


# =============== 캐시: 목록/썸네일 ===============
@st.cache_data(show_spinner=False)
def list_images(root: str, recursive: bool = False):
    """지정된 폴더에서 이미지 파일을 찾습니다."""
    patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
                "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
    if USE_HEIC:
        patterns += ["*.heic","*.heif","*.HEIC","*.HEIF"]
    paths = []
    root_path = Path(root)
    if not root_path.exists():
        return []
    
    if recursive:
        for pat in patterns:
            paths += list(root_path.rglob(pat))
    else:
        for pat in patterns:
            paths += list(root_path.glob(pat))
    
    paths = [p for p in paths if p.is_file()]
    return [str(p) for p in sorted(set(paths))]

@st.cache_data(show_spinner=False)
def load_thumbnail(path: str, max_side: int = 384):
    """썸네일 이미지를 생성합니다."""
    img = imread_any(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# =============== 시각화 함수 ===============
def plot_score_distribution(scores_dict, mode):
    """점수 분포를 박스 플롯으로 시각화합니다."""
    data = []
    for path, scores in scores_dict.items():
        if mode == "simple":
            data.append({
                'Image': Path(path).name,
                'Score': scores['score']
            })
        else:
            data.append({
                'Image': Path(path).name,
                'Sharp': scores['sharp_score'],
                'Defocus': scores['defocus_score'],
                'Motion': scores['motion_score']
            })
    
    df = pd.DataFrame(data)
    
    if mode == "simple":
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['Score'], name='점수', marker_color='blue'))
        fig.update_layout(title='점수 분포', yaxis_title='Score', height=400)
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df['Sharp'], name='Sharp', marker_color='green'))
        fig.add_trace(go.Box(y=df['Defocus'], name='Defocus', marker_color='orange'))
        fig.add_trace(go.Box(y=df['Motion'], name='Motion', marker_color='red'))
        fig.update_layout(title='점수 분포 (Box Plot)', yaxis_title='Score', showlegend=True, height=400)
    
    return fig

def plot_feature_radar(features_dict):
    """특징을 레이더 차트로 시각화합니다."""
    categories = list(features_dict.keys())
    values = list(features_dict.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Features'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=400,
        title='정규화된 특징 레이더 차트'
    )
    
    return fig

def plot_class_distribution(labels_dict):
    """클래스 분포를 파이 차트로 시각화합니다."""
    df = pd.Series(labels_dict).value_counts()
    
    fig = px.pie(
        values=df.values,
        names=df.index,
        title='클래스 분포',
        color=df.index,
        color_discrete_map={'sharp': 'green', 'defocus': 'orange', 'motion': 'red', 
                           '선명 ✅': 'green', '아웃포커스 🌫️': 'orange', '모션블러 📸': 'red'}
    )
    
    return fig


# =============== Streamlit UI ===============
st.set_page_config(page_title="통합 이미지 품질 검사", layout="wide", page_icon="📷")

# 상단 모드 선택
st.title("📷 이미지 품질 검사 도구")

mode_col1, mode_col2 = st.columns([3, 1])
with mode_col1:
    st.markdown("하나의 도구로 간단한 검사부터 전문적인 분석까지 모두 가능합니다")
with mode_col2:
    app_mode = st.selectbox(
        "사용 모드",
        ["🎯 간단 모드", "⚙️ 고급 모드"],
        help="간단 모드는 빠르고 쉬운 검사를, 고급 모드는 상세한 분석을 제공합니다"
    )

is_simple = (app_mode == "🎯 간단 모드")

st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.header("📁 폴더 설정")
    
    if is_simple:
        # 간단 모드: 빠른 선택
        desktop_path = str(Path.home() / "Desktop")
        pictures_path = str(Path.home() / "Pictures")
        downloads_path = str(Path.home() / "Downloads")
        
        quick_select = st.radio(
            "빠른 선택:",
            ["직접 입력", "바탕화면", "사진", "다운로드"],
            index=1
        )
        
        if quick_select == "바탕화면":
            root = desktop_path
        elif quick_select == "사진":
            root = pictures_path
        elif quick_select == "다운로드":
            root = downloads_path
        else:
            root = st.text_input("폴더 경로", value=desktop_path)
    else:
        # 고급 모드: 직접 입력
        root = st.text_input(
            "이미지 폴더 경로",
            value=str(Path.home() / "Desktop")
        )
    
    st.caption(f"📂 {root}")
    recursive = st.checkbox("하위 폴더 포함", value=False)
    
    st.divider()
    
    if is_simple:
        # 간단 모드 설정
        st.header("⚙️ 검사 기준")
        quality_threshold = st.slider(
            "선명 기준 점수",
            30, 80, 60, 5,
            help="이 점수 이상이면 선명으로 판정됩니다"
        )
        show_filter = st.selectbox("보기", ["전체", "선명한 사진만", "흐린 사진만"], index=0)
    else:
        # 고급 모드 설정
        with st.expander("⚙️ 처리 옵션"):
            long_side = st.slider("분석용 리사이즈(긴 변)", 640, 2048, 1024, 64)
            tiles = st.slider("타일 수 (NxN)", 2, 6, 4, 1)
            use_multiprocessing = st.checkbox("멀티프로세싱 사용", value=True)
            max_workers = st.slider("워커 수", 1, cpu_count(), min(cpu_count(), 8))
            use_cnn_model = st.checkbox("CNN 모델 사용 (실험적)", value=False, disabled=not USE_CNN)
        
        with st.expander("🎚️ Sharp 가중치"):
            w_sharp_vol = st.slider("VoL", 0.0, 1.0, 0.30, 0.01)
            w_sharp_ten = st.slider("Tenengrad", 0.0, 1.0, 0.25, 0.01)
            w_sharp_hfr = st.slider("HighFreqRatio", 0.0, 1.0, 0.20, 0.01)
            w_sharp_esw = st.slider("EdgeSpread(역)", 0.0, 1.0, 0.15, 0.01)
            w_sharp_slope = st.slider("RadialSlope(역)", 0.0, 1.0, 0.10, 0.01)
        
        with st.expander("🎚️ Defocus 가중치"):
            w_def_esw = st.slider("EdgeSpread", 0.0, 1.0, 0.40, 0.01)
            w_def_vol = st.slider("VoL(역)", 0.0, 1.0, 0.25, 0.01)
            w_def_slope = st.slider("RadialSlope(역)", 0.0, 1.0, 0.25, 0.01)
            w_def_aniso = st.slider("Anisotropy(역)", 0.0, 1.0, 0.10, 0.01)
        
        with st.expander("🎚️ Motion 가중치"):
            w_mot_aniso = st.slider("Anisotropy", 0.0, 1.0, 0.60, 0.01)
            w_mot_strat = st.slider("StructureTensor", 0.0, 1.0, 0.30, 0.01)
            w_mot_volinv = st.slider("VoL(역)", 0.0, 1.0, 0.10, 0.01)
        
        with st.expander("🔍 분류/필터 옵션"):
            min_sharp = st.slider("선명 최소 스코어", 0.0, 1.0, 0.35, 0.01)
            min_def = st.slider("아웃포커스 최소 스코어", 0.0, 1.0, 0.35, 0.01)
            min_mot = st.slider("모션 최소 스코어", 0.0, 1.0, 0.35, 0.01)
            show_pred = st.selectbox("미리보기 필터", ["모두", "선명", "아웃포커스", "모션블러"], index=0)

# 파라미터 설정
if not is_simple:
    params = dict(
        long_side=long_side,
        w_sharp_vol=w_sharp_vol, w_sharp_ten=w_sharp_ten, w_sharp_hfr=w_sharp_hfr,
        w_sharp_esw=w_sharp_esw, w_sharp_slope=w_sharp_slope,
        w_def_esw=w_def_esw, w_def_vol=w_def_vol, w_def_slope=w_def_slope, w_def_aniso=w_def_aniso,
        w_mot_aniso=w_mot_aniso, w_mot_strat=w_mot_strat, w_mot_volinv=w_mot_volinv,
    )
else:
    params = {}

# CNN 모델 초기화
if not is_simple and "cnn_model" not in st.session_state and use_cnn_model:
    st.session_state["cnn_model"] = BlurClassifierCNN()

# 메인 영역
if is_simple:
    # ========== 간단 모드 UI ==========
    tab1, tab2, tab3 = st.tabs(["🔍 검사 시작", "📊 결과 보기", "💡 도움말"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("1️⃣ 사진 검사 시작하기")
            if st.button("🔍 검사 시작", type="primary", use_container_width=True):
                paths = list_images(root, recursive=recursive)
                
                if len(paths) == 0:
                    st.error(f"❌ '{root}' 폴더에서 이미지를 찾을 수 없습니다.")
                else:
                    st.success(f"✅ {len(paths)}장의 사진을 찾았습니다!")
                    
                    # 처리
                    if len(paths) > 10:
                        results = batch_process_images(paths, "simple", 4, params, max_workers=min(cpu_count(), 8))
                    else:
                        results = {}
                        progress_bar = st.progress(0)
                        for i, p in enumerate(paths):
                            img = imread_any(p)
                            if img is not None:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                results[p] = check_sharpness_simple(gray)
                            progress_bar.progress((i + 1) / len(paths))
                        progress_bar.empty()
                    
                    st.session_state["paths"] = paths
                    st.session_state["results"] = results
                    st.session_state["mode"] = "simple"
                    
                    # 통계
                    sharp_count = sum(1 for r in results.values() if r["score"] > quality_threshold)
                    blur_count = len(results) - sharp_count
                    
                    st.balloons()
                    st.success("🎉 검사 완료!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("전체", f"{len(results)}장")
                    with col_b:
                        st.metric("선명", f"{sharp_count}장", delta="✅")
                    with col_c:
                        st.metric("흐림", f"{blur_count}장", delta="⚠️")
        
        with col2:
            st.subheader("📌 안내")
            st.info("""
            **사용 방법:**
            
            1. 왼쪽에서 폴더 선택
            2. '검사 시작' 버튼 클릭
            3. '결과 보기' 탭에서 확인
            
            **팁:**
            - 사진이 많으면 시간이 걸립니다
            - 점수가 높을수록 선명합니다
            - 60점이 기본 기준입니다
            """)
    
    with tab2:
        st.subheader("2️⃣ 검사 결과")
        
        if "results" not in st.session_state or len(st.session_state["results"]) == 0:
            st.warning("⚠️ 먼저 '검사 시작' 탭에서 사진을 검사해주세요.")
        else:
            results = st.session_state["results"]
            paths = st.session_state["paths"]
            
            # 필터링
            filtered_paths = []
            for p in paths:
                if p not in results:
                    continue
                r = results[p]
                
                if show_filter == "선명한 사진만" and r["score"] <= quality_threshold:
                    continue
                if show_filter == "흐린 사진만" and r["score"] > quality_threshold:
                    continue
                
                filtered_paths.append(p)
            
            st.write(f"**표시 중:** {len(filtered_paths)}장")
            
            # 정렬 및 페이지네이션
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                sort_by = st.selectbox("정렬", ["점수 높은 순", "점수 낮은 순", "파일명 순"], index=0)
            with col2:
                per_page = st.selectbox("페이지당", [12, 24, 48], index=0)
            with col3:
                page = st.number_input("페이지", min_value=1, value=1, step=1)
            
            if sort_by == "점수 높은 순":
                filtered_paths.sort(key=lambda p: results[p]["score"], reverse=True)
            elif sort_by == "점수 낮은 순":
                filtered_paths.sort(key=lambda p: results[p]["score"])
            else:
                filtered_paths.sort()
            
            start = (page - 1) * per_page
            end = min(start + per_page, len(filtered_paths))
            page_paths = filtered_paths[start:end]
            
            st.divider()
            
            # 그리드
            cols = st.columns(4)
            for i, p in enumerate(page_paths):
                col = cols[i % 4]
                r = results[p]
                
                with col:
                    thumb = load_thumbnail(p, max_side=300)
                    if thumb is not None:
                        st.image(thumb, use_container_width=True)
                    
                    score = r["score"]
                    if score > quality_threshold:
                        st.success(f"**{r['type']}**")
                        st.caption(f"점수: {score}")
                    else:
                        st.warning(f"**{r['type']}**")
                        st.caption(f"점수: {score}")
                    
                    st.caption(f"📁 {Path(p).name[:25]}")
                    
                    with st.expander("상세"):
                        st.write(f"선명도: {r['laplacian']}")
                        st.write(f"엣지: {r['edge']}")
                        st.write(f"방향성: {r['direction']}")
            
            st.divider()
            
            # 일괄 작업
            st.subheader("3️⃣ 사진 정리하기")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 CSV 저장", use_container_width=True):
                    rows = []
                    for p in paths:
                        if p in results:
                            r = results[p]
                            rows.append({
                                "파일명": Path(p).name,
                                "경로": p,
                                "점수": r["score"],
                                "상태": r["type"],
                                "품질": r["quality"]
                            })
                    df = pd.DataFrame(rows)
                    csv_path = Path(root) / "이미지_검사_결과.csv"
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    st.success(f"✅ 저장: {csv_path}")
            
            with col2:
                if st.button("📁 흐린 사진 이동", use_container_width=True):
                    blur_folder = Path(root) / "흐린_사진"
                    blur_folder.mkdir(exist_ok=True)
                    moved = 0
                    for p in paths:
                        if p in results and results[p]["score"] <= quality_threshold:
                            try:
                                shutil.move(p, blur_folder / Path(p).name)
                                moved += 1
                            except:
                                pass
                    st.success(f"✅ {moved}장을 이동했습니다!")
            
            with col3:
                if st.button("🗑️ 흐린 사진 삭제", use_container_width=True):
                    if "confirm_delete" not in st.session_state:
                        st.session_state["confirm_delete"] = False
                    
                    if not st.session_state["confirm_delete"]:
                        st.session_state["confirm_delete"] = True
                        st.warning("⚠️ 다시 클릭하면 삭제됩니다!")
                    else:
                        deleted = 0
                        for p in paths:
                            if p in results and results[p]["score"] <= quality_threshold:
                                try:
                                    os.remove(p)
                                    deleted += 1
                                except:
                                    pass
                        st.success(f"✅ {deleted}장을 삭제했습니다!")
                        st.session_state["confirm_delete"] = False
            
            # 분포 차트
            if len(results) > 0:
                st.divider()
                st.subheader("📊 점수 분포")
                fig = plot_score_distribution(results, "simple")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💡 간단 모드 도움말")
        st.markdown("""
        ## 📖 사용 가이드
        
        ### 이 도구는 무엇을 하나요?
        - 사진을 자동으로 검사해서 **선명한 사진**과 **흐린 사진**을 구분합니다
        - 흐린 사진을 찾아서 정리하거나 삭제할 수 있습니다
        
        ### 점수 의미
        - **0~100점** 사이로 계산됩니다
        - **60점 이상**: 선명 ✅
        - **60점 미만**: 흐림 ⚠️
        
        ### 흐림 타입
        - **선명 ✅**: 초점이 잘 맞은 사진
        - **아웃포커스 🌫️**: 초점이 안 맞아서 흐림
        - **모션블러 📸**: 움직여서 흐림
        
        ### ⚠️ 주의사항
        - **삭제는 되돌릴 수 없습니다!**
        - 처음엔 "이동"으로 확인하세요
        - 중요한 사진은 백업하세요
        
        ### 💡 팁
        1. 작은 폴더로 먼저 테스트
        2. "점수 낮은 순"으로 정렬하면 흐린 사진 빠르게 확인
        3. 점수 기준을 조절해서 사용
        4. CSV로 저장하면 엑셀에서도 확인 가능
        
        ---
        
        **더 자세한 분석이 필요하다면?**
        → 상단에서 "⚙️ 고급 모드"로 전환하세요!
        """)

else:
    # ========== 고급 모드 UI ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "🖼️ 이미지 라벨링", "📈 통계 분석", "⚙️ 도움말"])
    
    with tab1:
        st.header("📊 분석 대시보드")
        
        paths = list_images(root, recursive=recursive)
        st.metric("총 이미지 수", len(paths))
        
        if len(paths) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚀 전체 이미지 분석 시작", type="primary"):
                    with st.spinner("분석 중..."):
                        if use_multiprocessing and len(paths) > 10:
                            results = batch_process_images(paths, "advanced", tiles, params, max_workers)
                        else:
                            results = {}
                            progress_bar = st.progress(0)
                            for i, p in enumerate(paths):
                                img = imread_any(p)
                                if img is not None:
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    results[p] = compute_scores_advanced(gray, tiles, params)
                                progress_bar.progress((i + 1) / len(paths))
                            progress_bar.empty()
                        
                        st.session_state["scores"] = results
                        st.session_state["mode"] = "advanced"
                        st.success(f"✅ {len(results)}개 이미지 분석 완료!")
            
            with col2:
                if "scores" in st.session_state and len(st.session_state["scores"]) > 0:
                    st.metric("분석 완료", f"{len(st.session_state['scores'])}장")
            
            with col3:
                if "labels" in st.session_state and len(st.session_state["labels"]) > 0:
                    st.metric("라벨링 완료", f"{len(st.session_state['labels'])}장")
            
            if "scores" in st.session_state and len(st.session_state["scores"]) > 0:
                st.subheader("📉 점수 분포 분석")
                fig = plot_score_distribution(st.session_state["scores"], "advanced")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🖼️ 이미지 라벨링")
        
        if "labels" not in st.session_state:
            st.session_state["labels"] = {}
        if "scores" not in st.session_state:
            st.session_state["scores"] = {}
        
        col1, col2 = st.columns(2)
        with col1:
            per_page = st.selectbox("페이지당 썸네일 수", [12, 24, 48], index=1)
        with col2:
            page = st.number_input("페이지(1부터)", min_value=1, value=1, step=1)
        
        start = (page-1)*per_page
        end = min(start+per_page, len(paths))
        page_paths = paths[start:end]
        
        grid_cols = st.columns(4)
        for i, p in enumerate(page_paths):
            col = grid_cols[i % 4]
            with col:
                thumb = load_thumbnail(p, max_side=384)
                if thumb is not None:
                    st.image(thumb, use_column_width=True)
                
                if p in st.session_state["scores"]:
                    S = st.session_state["scores"][p]
                else:
                    img = imread_any(p)
                    if img is None:
                        st.caption("읽기 실패")
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    S = compute_scores_advanced(gray, tiles=tiles, params=params)
                    st.session_state["scores"][p] = S
                
                sharp_s = S["sharp_score"]
                def_s = S["defocus_score"]
                mot_s = S["motion_score"]
                
                # CNN 예측
                cnn_pred = None
                if use_cnn_model and "cnn_model" in st.session_state:
                    img_cnn = imread_any(p)
                    if img_cnn is not None:
                        cnn_pred = st.session_state["cnn_model"].predict(img_cnn)
                
                preds = [("sharp", sharp_s), ("defocus", def_s), ("motion", mot_s)]
                pred = max(preds, key=lambda x: x[1])[0]
                
                if pred == "sharp" and sharp_s < min_sharp:
                    pred = "defocus" if def_s >= max(min_def, mot_s) else "motion"
                if pred == "defocus" and def_s < min_def:
                    pred = "sharp" if sharp_s >= max(min_sharp, mot_s) else "motion"
                if pred == "motion" and mot_s < min_mot:
                    pred = "sharp" if sharp_s >= max(min_sharp, def_s) else "defocus"
                
                if cnn_pred:
                    cnn_class = max(cnn_pred, key=cnn_pred.get)
                    st.caption(f"CNN: {cnn_class} ({cnn_pred[cnn_class]:.2f})")
                
                if show_pred != "모두":
                    need = {"선명":"sharp", "아웃포커스":"defocus", "모션블러":"motion"}[show_pred]
                    if pred != need:
                        continue
                
                current = st.session_state["labels"].get(p, pred)
                new_label = st.selectbox(
                    label=f"{Path(p).name[:20]}...\nS:{sharp_s:.2f} D:{def_s:.2f} M:{mot_s:.2f}",
                    options=["sharp","defocus","motion"],
                    index=["sharp","defocus","motion"].index(current),
                    key=f"sel_{p}"
                )
                st.session_state["labels"][p] = new_label
                
                if st.button("📊 특징 보기", key=f"feat_{p}"):
                    if "normalized" in S:
                        fig = plot_feature_radar(S["normalized"])
                        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            bulk_label = st.selectbox("페이지 전체 라벨", ["(선택 없음)","sharp","defocus","motion"])
        with col2:
            if st.button("일괄 적용"):
                if bulk_label != "(선택 없음)":
                    for p in page_paths:
                        if p in st.session_state["scores"]:
                            st.session_state["labels"][p] = bulk_label
                    st.success("일괄 적용 완료!")
        
        with col3:
            if st.button("💾 CSV 저장"):
                rows = []
                for p in paths:
                    if p in st.session_state["scores"]:
                        lab = st.session_state["labels"].get(p, "sharp")
                        sc = st.session_state["scores"][p]
                        rows.append({
                            "path": p, "label": lab,
                            "sharp_score": round(sc["sharp_score"],4),
                            "defocus_score": round(sc["defocus_score"],4),
                            "motion_score": round(sc["motion_score"],4),
                        })
                df = pd.DataFrame(rows)
                out_csv = Path(root) / "labels.csv"
                df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                st.success(f"저장: {out_csv}")
        
        with col4:
            move_or_copy = st.selectbox("내보내기 방식", ["copy","move"])
        
        with col5:
            if st.button("📦 학습셋 내보내기"):
                out_root = Path(root) / "train"
                (out_root / "sharp").mkdir(parents=True, exist_ok=True)
                (out_root / "defocus").mkdir(parents=True, exist_ok=True)
                (out_root / "motion").mkdir(parents=True, exist_ok=True)
                
                n_done = 0
                progress = st.progress(0)
                labeled_paths = [p for p in paths if p in st.session_state["labels"]]
                
                for idx, p in enumerate(labeled_paths):
                    lab = st.session_state["labels"][p]
                    dst = out_root / lab / Path(p).name
                    try:
                        if move_or_copy == "copy":
                            shutil.copy2(p, dst)
                        else:
                            shutil.move(p, dst)
                        n_done += 1
                    except:
                        pass
                    progress.progress((idx + 1) / len(labeled_paths))
                
                progress.empty()
                st.success(f"✅ {n_done}개 파일 → {out_root}")
    
    with tab3:
        st.header("📈 통계 분석")
        
        if "labels" in st.session_state and len(st.session_state["labels"]) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("클래스 분포")
                fig = plot_class_distribution(st.session_state["labels"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("라벨 통계")
                ser = pd.Series(st.session_state["labels"]).value_counts()
                df_stats = ser.to_frame("개수")
                df_stats["비율(%)"] = (df_stats["개수"] / df_stats["개수"].sum() * 100).round(2)
                st.dataframe(df_stats, use_container_width=True)
            
            if "scores" in st.session_state:
                st.subheader("라벨별 점수 분포")
                data = []
                for path in st.session_state["labels"].keys():
                    if path in st.session_state["scores"]:
                        label = st.session_state["labels"][path]
                        scores = st.session_state["scores"][path]
                        data.append({
                            "Label": label,
                            "Sharp Score": scores["sharp_score"],
                            "Defocus Score": scores["defocus_score"],
                            "Motion Score": scores["motion_score"]
                        })
                
                if data:
                    df = pd.DataFrame(data)
                    
                    st.subheader("클래스별 평균 점수")
                    avg_scores = df.groupby("Label")[["Sharp Score", "Defocus Score", "Motion Score"]].mean()
                    st.dataframe(avg_scores.round(3), use_container_width=True)
                    
                    st.subheader("점수 상관관계")
                    fig = px.scatter_matrix(
                        df,
                        dimensions=["Sharp Score", "Defocus Score", "Motion Score"],
                        color="Label",
                        color_discrete_map={'sharp': 'green', 'defocus': 'orange', 'motion': 'red'},
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("라벨링된 이미지가 없습니다.")
    
    with tab4:
        st.header("⚙️ 고급 모드 도움말")
        st.markdown("""
        ## 📖 고급 모드 가이드
        
        ### 특징
        - 3-클래스 분류 (선명/아웃포커스/모션블러)
        - 가중치 조절 가능
        - 멀티프로세싱 지원
        - CNN 모델 통합 (실험적)
        - 학습 데이터셋 생성
        
        ### 워크플로우
        1. 대시보드에서 전체 분석
        2. 라벨링 탭에서 수동 보정
        3. 통계 분석으로 데이터 검증
        4. 학습셋으로 내보내기
        
        ### 성능 최적화
        - 멀티프로세싱 활성화 (권장)
        - 리사이즈 크기 조정 (896-1280)
        - 타일 수 조정 (3-5)
        
        ### 지표 설명
        - **VoL**: 라플라시안 분산 (선명도)
        - **Tenengrad**: 그라디언트 에너지
        - **HighFreqRatio**: 고주파 비율
        - **EdgeSpreadWidth**: 엣지 확산 폭
        - **Anisotropy**: 방향성 (모션블러)
        - **StructureTensor**: 구조 타원성
        
        ---
        
        **빠른 검사가 필요하다면?**
        → 상단에서 "🎯 간단 모드"로 전환하세요!
        """)

# 푸터
st.divider()
st.caption("💡 팁: 용도에 맞는 모드를 선택하세요 | 간단 모드 = 빠른 검사 | 고급 모드 = 상세 분석")