# -*- coding: utf-8 -*-
import os, glob, math, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

# =========================
# (선택) HEIC 지원 설정
# =========================
USE_HEIC = False
try:
    import pillow_heif
    USE_HEIC = True
except Exception:
    USE_HEIC = False  # pillow-heif 미설치 시 자동 비활성


# =========================
# 파일 로딩 (한글/공백 경로 안전 + HEIC 옵션)
# =========================
def imread_any(path):
    p = str(path)
    ext = p.lower().split(".")[-1]
    if USE_HEIC and ext in ("heic", "heif"):
        heif = pillow_heif.read_heif(p)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data


# =========================
# 저수준 특징들
# =========================
def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))

def highfreq_ratio(gray, cutoff=0.1):
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

def radial_spectrum_slope(gray, cutoff=0.6):
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
    return float(slope)  # 더 음수일수록 defocus ↑

def anisotropy_index(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)  # 0~2π
    nbins = 18
    hist, _ = np.histogram(ang, bins=nbins, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return float(np.std(hist))  # 모션블러 ↑ (특정 방향 치우침)

def structure_tensor_ratio(gray):
    # 구조텐서로 방향성(타원성) 측정: (λ1 - λ2)/(λ1 + λ2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), 1.0)
    Jyy = cv2.GaussianBlur(gy*gy, (0,0), 1.0)
    Jxy = cv2.GaussianBlur(gx*gy, (0,0), 1.0)
    # 지역 평균으로 대표값
    l1 = 0.5*((Jxx+Jyy) + np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    l2 = 0.5*((Jxx+Jyy) - np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    num = (l1 - l2)
    den = (l1 + l2 + 1e-8)
    ratio = np.mean(num/den)
    return float(ratio)  # 모션블러 ↑ (방향 길쭉함)

def edge_spread_width(gray, sample_edges=150):
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


# =========================
# 타일링 특징 (하위 퍼센타일로 보수적 집계)
# =========================
def tile_features(gray, tiles=4):
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
    # defocus는 취약 타일(하위 퍼센타일)이 신호 → 20% 사용
    # motion 방향성 지표는 상위가 신호 → 80% 사용
    feats = {
        "vol_p20": p(vols,20), "ten_p20": p(tens,20), "hfr_p20": p(hfrs,20),
        "esw_p80": p(esws,80), # 엣지폭은 커질수록 defocus ↑ → 상위 퍼센타일
        "aniso_p80": p(anisos,80),
        "slope_p20": p(slopes,20),
        "strat_p80": p(strats,80),
    }
    return feats


# =========================
# 정규화 & 스코어링
# =========================
def norm_box(x, lo, hi, invert=False):
    x = float(x)
    v = (x - lo) / (hi - lo + 1e-8)
    v = min(max(v, 0.0), 1.0)
    return 1.0 - v if invert else v

def compute_scores(gray, tiles, params):
    # 리사이즈(긴 변)
    H, W = gray.shape
    long_side = params["long_side"]
    if max(H, W) > long_side:
        s = long_side / max(H, W)
        gray = cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    F = tile_features(gray, tiles=tiles)

    # 노멀라이즈 (경험적 범위, 필요시 사이드바에서 조정 가능하게 해도 됨)
    vol_n   = norm_box(F["vol_p20"],   50,   600,  invert=False)  # 선명↑
    ten_n   = norm_box(F["ten_p20"],   1.0,  12.0, invert=False)  # 선명↑
    hfr_n   = norm_box(F["hfr_p20"],   0.02, 0.35, invert=False)  # 선명↑
    esw_n   = norm_box(F["esw_p80"],   1.0,  6.0,  invert=True)   # 선명↑(엣지 폭 작을수록)
    slope_n = norm_box(F["slope_p20"], -6.0, -0.5, invert=True)   # 선명↑(덜 음수)
    aniso_n = norm_box(F["aniso_p80"], 0.0,  0.12, invert=False)  # 모션↑
    strat_n = norm_box(F["strat_p80"], 0.02, 0.45, invert=False)  # 모션↑

    # 3-클래스 스코어
    sharp_score = (
        params["w_sharp_vol"] * vol_n +
        params["w_sharp_ten"] * ten_n +
        params["w_sharp_hfr"] * hfr_n +
        params["w_sharp_esw"] * esw_n +
        params["w_sharp_slope"] * slope_n
    )

    defocus_score = (
        params["w_def_esw"] * (1 - esw_n) +      # 엣지폭 클수록 defocus↑ → 선명 정규화의 보완
        params["w_def_vol"] * (1 - vol_n) +      # VoL 낮을수록 defocus↑
        params["w_def_slope"] * (1 - slope_n) +  # 더 음수일수록 defocus↑
        params["w_def_aniso"] * (1 - aniso_n)    # 등방성일수록 defocus↑
    )

    motion_score = (
        params["w_mot_aniso"] * aniso_n +
        params["w_mot_strat"] * strat_n +
        params["w_mot_volinv"] * (1 - vol_n)     # 흐릴수록(저 VoL) 모션 가능성↑
    )

    return {
        "features": F,
        "sharp_score": float(sharp_score),
        "defocus_score": float(defocus_score),
        "motion_score": float(motion_score),
    }


# =========================
# 캐시: 이미지 목록 / 썸네일 / 특징
# =========================
@st.cache_data(show_spinner=False)
def list_images(root, recursive=False):
    patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
                "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
    if USE_HEIC:
        patterns += ["*.heic","*.heif","*.HEIC","*.HEIF"]
    paths = []
    if recursive:
        for pat in patterns:
            paths.extend(Path(root).rglob(pat))
    else:
        for pat in patterns:
            paths.extend(Path(root).glob(pat))
    paths = [p for p in paths if p.is_file()]
    return [str(p) for p in sorted(set(paths))]

@st.cache_data(show_spinner=False)
def load_thumbnail(path, max_side=384):
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
def compute_scores_cached(path, tiles, params):
    img = imread_any(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return compute_scores(gray, tiles=tiles, params=params)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="3-클래스 분류/라벨링", layout="wide")
st.title("📷 모션블러 / 아웃포커스 / 선명 — 3-클래스 분류 + 수동 라벨링 툴")

with st.sidebar:
    st.header("폴더 & 스캔")
    root = st.text_input("이미지 폴더 경로", value=str(Path.home() / "Pictures"))
    recursive = st.checkbox("하위 폴더까지 포함", value=False)
    long_side = st.slider("분석용 리사이즈(긴 변)", 640, 2048, 1024, 64)

    st.header("타일 / 가중치")
    tiles = st.slider("타일 수 (NxN)", 2, 6, 4, 1)

    st.caption("— Sharp(선명) 가중치 —")
    w_sharp_vol   = st.slider("VoL", 0.0, 1.0, 0.30, 0.01)
    w_sharp_ten   = st.slider("Tenengrad", 0.0, 1.0, 0.25, 0.01)
    w_sharp_hfr   = st.slider("HighFreqRatio", 0.0, 1.0, 0.20, 0.01)
    w_sharp_esw   = st.slider("EdgeSpread(역)", 0.0, 1.0, 0.15, 0.01)
    w_sharp_slope = st.slider("RadialSlope(역)", 0.0, 1.0, 0.10, 0.01)

    st.caption("— Defocus(아웃포커스) 가중치 —")
    w_def_esw   = st.slider("EdgeSpread", 0.0, 1.0, 0.40, 0.01)
    w_def_vol   = st.slider("VoL(역)", 0.0, 1.0, 0.25, 0.01)
    w_def_slope = st.slider("RadialSlope(역)", 0.0, 1.0, 0.25, 0.01)
    w_def_aniso = st.slider("Anisotropy(역)", 0.0, 1.0, 0.10, 0.01)

    st.caption("— Motion(모션블러) 가중치 —")
    w_mot_aniso   = st.slider("Anisotropy", 0.0, 1.0, 0.60, 0.01)
    w_mot_strat   = st.slider("StructureTensor", 0.0, 1.0, 0.30, 0.01)
    w_mot_volinv  = st.slider("VoL(역)", 0.0, 1.0, 0.10, 0.01)

    st.header("분류/필터 옵션")
    min_sharp = st.slider("선명 최소 스코어", 0.0, 1.0, 0.35, 0.01)
    min_def   = st.slider("아웃포커스 최소 스코어", 0.0, 1.0, 0.35, 0.01)
    min_mot   = st.slider("모션 최소 스코어", 0.0, 1.0, 0.35, 0.01)
    show_pred = st.selectbox("미리보기 필터", ["모두", "선명", "아웃포커스", "모션블러"], index=0)

params = dict(
    long_side=long_side,
    w_sharp_vol=w_sharp_vol, w_sharp_ten=w_sharp_ten, w_sharp_hfr=w_sharp_hfr,
    w_sharp_esw=w_sharp_esw, w_sharp_slope=w_sharp_slope,
    w_def_esw=w_def_esw, w_def_vol=w_def_vol, w_def_slope=w_def_slope, w_def_aniso=w_def_aniso,
    w_mot_aniso=w_mot_aniso, w_mot_strat=w_mot_strat, w_mot_volinv=w_mot_volinv,
)

# 이미지 목록
paths = list_images(root, recursive=recursive)
st.write(f"**총 이미지**: {len(paths)}")

# 페이지네이션
cols = st.columns(2)
with cols[0]:
    per_page = st.selectbox("페이지당 썸네일 수", [12, 24, 48], index=1)
with cols[1]:
    page = st.number_input("페이지(1부터)", min_value=1, value=1, step=1)
start = (page-1)*per_page
end = min(start+per_page, len(paths))
page_paths = paths[start:end]

# 상태 저장: 라벨 딕셔너리
if "labels" not in st.session_state:
    st.session_state["labels"] = {}  # path -> "sharp|defocus|motion"
if "scores" not in st.session_state:
    st.session_state["scores"] = {}  # path -> dict(scores)

# 썸네일 그리드
grid_cols = st.columns(4)
for i, p in enumerate(page_paths):
    col = grid_cols[i % 4]
    with col:
        thumb = load_thumbnail(p, max_side=384)
        if thumb is not None:
            st.image(thumb, use_column_width=True)

        # 스코어/예측
        S = compute_scores_cached(p, tiles=tiles, params=params)
        if S is None:
            st.caption("읽기 실패")
            continue

        sharp_s = S["sharp_score"]
        def_s   = S["defocus_score"]
        mot_s   = S["motion_score"]

        # argmax 분류 + 최소 스코어 적용
        preds = [("sharp", sharp_s), ("defocus", def_s), ("motion", mot_s)]
        pred = max(preds, key=lambda x: x[1])[0]
        if pred == "sharp" and sharp_s < min_sharp:
            pred = "defocus" if def_s >= max(min_def, mot_s) else "motion"
        if pred == "defocus" and def_s < min_def:
            pred = "sharp" if sharp_s >= max(min_sharp, mot_s) else "motion"
        if pred == "motion" and mot_s < min_mot:
            pred = "sharp" if sharp_s >= max(min_sharp, def_s) else "defocus"

        # 필터링
        if show_pred != "모두":
            need = {"선명":"sharp", "아웃포커스":"defocus", "모션블러":"motion"}[show_pred]
            if pred != need:
                continue

        # 저장
        st.session_state["scores"][p] = dict(sharp=sharp_s, defocus=def_s, motion=mot_s)

        # 수동 라벨 드롭다운
        current = st.session_state["labels"].get(p, pred)
        new_label = st.selectbox(
            label=f"{Path(p).name}\nS:{sharp_s:.2f} D:{def_s:.2f} M:{mot_s:.2f}",
            options=["sharp","defocus","motion"],
            index=["sharp","defocus","motion"].index(current),
            key=f"sel_{p}"
        )
        st.session_state["labels"][p] = new_label

st.divider()

# 일괄 적용/저장 영역
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    bulk_label = st.selectbox("페이지 전체 라벨", ["(선택 없음)","sharp","defocus","motion"])
with c2:
    if st.button("이 페이지에 일괄 적용"):
        for p in page_paths:
            if p in st.session_state["scores"]:
                st.session_state["labels"][p] = bulk_label if bulk_label != "(선택 없음)" else st.session_state["labels"].get(p, "sharp")
        st.success("이 페이지의 항목에 일괄 적용 완료")

with c3:
    if st.button("CSV 저장 (labels.csv)"):
        rows = []
        for p in paths:
            if p in st.session_state["scores"]:
                lab = st.session_state["labels"].get(p, "sharp")
                sc = st.session_state["scores"][p]
                rows.append({
                    "path": p, "label": lab,
                    "sharp_score": round(sc["sharp"],4),
                    "defocus_score": round(sc["defocus"],4),
                    "motion_score": round(sc["motion"],4),
                })
        df = pd.DataFrame(rows)
        out_csv = Path(root) / "labels.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        st.success(f"저장: {out_csv}")

with c4:
    move_or_copy = st.selectbox("내보내기 방식", ["copy","move"])
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
            if move_or_copy == "copy":
                try:
                    shutil.copy2(p, dst)
                    n_done += 1
                except Exception as e:
                    st.warning(f"복사 실패: {p} -> {dst} ({e})")
            else:
                try:
                    shutil.move(p, dst)
                    n_done += 1
                except Exception as e:
                    st.warning(f"이동 실패: {p} -> {dst} ({e})")

        st.success(f"학습셋 내보내기 완료 ({move_or_copy}): {n_done}개 → {out_root}")

# 하단: 현재 라벨링 요약
st.subheader("라벨 요약")
if st.session_state["labels"]:
    ser = pd.Series(st.session_state["labels"]).value_counts()
    st.write(ser.to_frame("count"))
else:
    st.write("아직 라벨이 없습니다.")
