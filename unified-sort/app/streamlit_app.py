# -*- coding: utf-8 -*-
"""
Streamlit UI for Unified Image Quality Classifier
- 간단 모드/고급 모드 모두 포함
- 고해상도 보기, 자동 태깅/파일명 변경, 유사도(pHash) 묶기, RW2->JPG 변환, 휴지통 삭제(선택) 포함
"""

from __future__ import annotations
import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2
import streamlit as st

# 로컬 패키지
import unified_sort as us  # list_images/batch_analyze/load_thumbnail/... 등을 사용

# 선택적 의존성
try:
    from send2trash import send2trash  # 휴지통으로 이동
except Exception:
    send2trash = None

try:
    import rawpy  # RW2 등 RAW 처리
except Exception:
    rawpy = None

try:
    # 권장: imageio.v3
    import imageio.v3 as iio
except Exception:
    iio = None

try:
    import pillow_heif  # type: ignore
    _USE_HEIC = True
except Exception:
    pillow_heif = None
    _USE_HEIC = False


# ---------------------------------------------------------------------
# 이미지 로딩(폴백 포함)
# ---------------------------------------------------------------------
def _imread_any_local(path: str):
    """HEIC/HEIF 지원 + 일반 이미지(CV2) 로딩 폴백."""
    p = str(path)
    ext = p.lower().split(".")[-1]
    if _USE_HEIC and ext in ("heic", "heif"):
        try:
            heif = pillow_heif.read_heif(p)
            from PIL import Image as _PILImage
            img = _PILImage.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data

def imread_any_compat(path: str):
    """unified_sort.imread_any가 있으면 사용, 없으면 로컬 폴백. 실패 시 None."""
    try:
        fn = getattr(us, "imread_any", None)
        if callable(fn):
            try:
                return fn(path)
            except Exception:
                return _imread_any_local(path)
        return _imread_any_local(path)
    except Exception:
        return None


# ---------------------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------------------
def make_widget_key(prefix: str, path: str) -> str:
    """path 기반 위젯 키."""
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

def show_modal(title: str, render_fn: Callable[[], None], width: str = "large"):
    """
    Streamlit 1.36+ : st.dialog(title, width=...) 데코레이터 기반 모달
    이전 버전        : 인라인 섹션으로 폴백
    """
    dlg = getattr(st, "dialog", None)
    if callable(dlg):
        @dlg(title, width=width)
        def _inner():
            render_fn()
        _inner()
    else:
        st.subheader(title)  # fallback
        render_fn()
        st.markdown("---")

def load_fullres(path: str, max_side: int | None = 2048):
    """
    원본 이미지를 로드(필요 시 리사이즈)하여 **RGB** ndarray 반환.
    실패 시 None.
    """
    try:
        img_bgr = imread_any_compat(path)
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception:
        return None


# pHash / Hamming (빠른 유사도 묶기)
def phash_from_gray(gray: np.ndarray, hash_size: int = 8) -> int:
    g = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    return sum(1 << i for (i, v) in enumerate(diff.flatten()) if v)

def hamming_dist(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# 고급 모드 성능: 파라미터-키 + 결과 캐시
def _params_key_for_cache(tiles: int, params: dict) -> str:
    key_dict = dict(params)
    key_dict["tiles"] = tiles
    blob = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def compute_scores_cached(path: str, tiles: int, params: dict, params_key: str) -> dict:
    """항상 us.batch_analyze로 1장만 분석하여 결과 반환(캐시)."""
    try:
        res = us.batch_analyze([path], mode="advanced", tiles=tiles, params=params, max_workers=1)
        return res.get(path, {})
    except Exception:
        return {}


# ---------------------------------------------------------------------
# 페이지/모드 설정
# ---------------------------------------------------------------------
st.set_page_config(page_title="통합 이미지 품질 검사", layout="wide", page_icon="📷")
st.title("📷 이미지 품질 검사 도구")

mode_col1, mode_col2 = st.columns([3, 1])
with mode_col1:
    st.markdown("하나의 도구로 간단한 검사부터 전문적인 분석까지 모두 가능합니다")
with mode_col2:
    app_mode = st.selectbox("사용 모드", ["🎯 간단 모드", "⚙️ 고급 모드"])
is_simple = (app_mode == "🎯 간단 모드")
st.markdown("---")

# ---- safe defaults (사이드바 슬라이더의 초기값으로도 사용) ----
use_hybrid = False
exif_on = True
face_on = True
face_alpha = 0.6
roi_free = False
dl_on = False
dl_weight = 0.5
dl_motion_bias = 0.0
dl_weights_path = ""
long_side = 1024
tiles = 4
max_workers = 8

# ---------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("📁 폴더 설정")
    if is_simple:
        desktop_path = str(Path.home() / "Desktop")
        pictures_path = str(Path.home() / "Pictures")
        downloads_path = str(Path.home() / "Downloads")
        quick_select = st.radio("빠른 선택:", ["직접 입력", "바탕화면", "사진", "다운로드"], index=1)
        if quick_select == "바탕화면":
            root = desktop_path
        elif quick_select == "사진":
            root = pictures_path
        elif quick_select == "다운로드":
            root = downloads_path
        else:
            root = st.text_input("폴더 경로", value=desktop_path)
    else:
        root = st.text_input("이미지 폴더 경로", value=str(Path.home() / "Desktop"))

    st.caption(f"📂 {root}")
    recursive = st.checkbox("하위 폴더 포함", value=False)
    st.divider()

    if is_simple:
        st.header("⚙️ 검사 기준")
        quality_threshold = st.slider("선명 기준 점수", 30, 80, 60, 5)
        show_filter = st.selectbox("보기", ["전체", "선명한 사진만", "흐린 사진만"], index=0)
    else:
        with st.expander("⚙️ 처리 옵션"):
            long_side = st.slider("분석 리사이즈(긴 변)", 640, 2048, long_side, 64)
            tiles = st.slider("타일 수 (NxN)", 2, 6, tiles, 1)
            max_workers = st.slider("워커 수", 1, 16, max_workers)

        with st.expander("🧪 하이브리드 옵션"):
            use_hybrid = st.checkbox("하이브리드 품질 분석(권장)", value=True)
            exif_on = st.checkbox("EXIF 보정 사용", value=exif_on)
            face_on = st.checkbox("인물(얼굴) 가중치 사용", value=face_on)
            face_alpha = st.slider("얼굴 가중치 알파", 0.0, 1.0, face_alpha, 0.05)
            roi_free = st.checkbox("ROI-free 보정", value=roi_free)
            dl_on = st.checkbox("딥러닝 NR-IQA 융합", value=True)
            dl_weight = st.slider("DL 가중치(0~1)", 0.0, 1.0, dl_weight, 0.05)
            dl_motion_bias = st.slider("DL 모션 바이어스", -0.5, 0.5, dl_motion_bias, 0.05)
            dl_weights_path = st.text_input("DL 가중치(선택)", value=dl_weights_path)


# ---------------------------------------------------------------------
# 간단 모드
# ---------------------------------------------------------------------
if is_simple:
    tab1, tab2 = st.tabs(["🔍 검사 시작", "📊 결과 보기"])

    # 검사 시작
    with tab1:
        if st.button("🔍 검사 시작", type="primary", use_container_width=True):
            paths = us.list_images(root, recursive=recursive)
            st.session_state["paths"] = paths
            if not paths:
                st.error("이미지를 찾을 수 없습니다.")
            else:
                # 간단 모드용 파라미터(DL은 기본 OFF 권장)
                params_simple = {
                    "long_side": long_side,
                    "exif_correction": exif_on,
                    "face_prior_enabled": face_on,
                    "face_prior_alpha": face_alpha,
                    "roi_free": roi_free,
                    "enable_dl_hybrid": False,
                }
                res = us.batch_analyze(paths, mode="simple", params=params_simple)
                st.session_state["results_simple"] = res
                st.success(f"✅ {len(res)}장 분석 완료")

    # 결과 보기
    with tab2:
        res: Dict[str, dict] = st.session_state.get("results_simple", {})
        paths: List[str] = st.session_state.get("paths", [])
        if not res:
            st.info("먼저 '검사 시작'을 눌러주세요.")
        else:
            # 필터링
            filtered: List[str] = []
            for p in paths:
                r = res.get(p)
                if not r:
                    continue
                score = r["score"]
                if show_filter == "선명한 사진만" and score <= quality_threshold:
                    continue
                if show_filter == "흐린 사진만" and score > quality_threshold:
                    continue
                filtered.append(p)

            st.write(f"**표시 중:** {len(filtered)}장")

            # 그리드
            cols = st.columns(4)
            for i, p in enumerate(filtered[:48]):
                col = cols[i % 4]
                with col:
                    thumb = us.load_thumbnail(p, max_side=320)
                    if thumb is not None:
                        col.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), use_container_width=True)

                    r = res[p]
                    st.caption(Path(p).name[:32])

                    score = r["score"]
                    if score > quality_threshold:
                        st.success(f"{r['type']} • {score}")
                    else:
                        st.warning(f"{r['type']} • {score}")

                    if col.button("🔎 고해상도", key=make_widget_key("zoom_simple", p)):
                        def _render():
                            big = load_fullres(p, max_side=2048)
                            if big is not None:
                                st.image(big, use_container_width=True)
                        show_modal(f"고해상도 · {Path(p).name}", _render, width="large")

            st.divider()

            # ------------------ 추가 정리 기능들 ------------------
            st.subheader("🛠️ 추가 정리 기능")

            # 1) 파일명 자동 태깅/변경
            with st.expander("✍️ 파일명 자동 태깅/변경"):
                st.caption("패턴 예시: `{stem}_{tag}` 또는 `{stem}_score{score}`")
                rule = st.text_input("파일명 패턴", value="{stem}_{tag}")
                tag_mode = st.selectbox("태그 기준", ["흐림/선명", "흐림 타입(모션/아웃)", "점수 구간(10점 단위)"])
                dry_run = st.checkbox("미리보기(실제 변경 안 함)", value=True, key="dry_simple")
                apply_scope = st.selectbox("대상 범위", ["현재 페이지에 표시된 사진만", "필터된 전체 사진"], index=0)

                if st.button("파일명 적용", key="apply_rename_simple"):
                    target = filtered[:48] if apply_scope.startswith("현재") else filtered
                    renamed = 0; conflict = 0; failed = 0
                    for p in target:
                        r = res[p]
                        score = r["score"]
                        if tag_mode == "흐림/선명":
                            tag = "sharp" if score > quality_threshold else "blur"
                        elif tag_mode == "흐림 타입(모션/아웃)":
                            tag = "sharp" if score > quality_threshold else ("motion" if r["type"].startswith("모션") else "defocus")
                        else:
                            tag = f"score{int(score//10)*10}"

                        stem = Path(p).stem
                        ext = Path(p).suffix
                        new_name = rule.format(stem=stem, tag=tag, score=int(score))
                        dst = Path(p).with_name(new_name + ext)

                        if dst.exists():
                            conflict += 1
                            continue
                        if not dry_run:
                            try:
                                os.rename(p, dst)
                                st.session_state["paths"] = [str(dst) if x == p else x for x in st.session_state.get("paths", [])]
                                renamed += 1
                            except Exception:
                                failed += 1
                    st.success(f"완료: {renamed}개, 이름 충돌: {conflict}개, 실패: {failed}개")

            # 2) 유사도 기반 묶기 (pHash)
            with st.expander("🧩 비슷한 사진 묶기(실험적, pHash)"):
                st.caption("간단한 perceptual hash 기반으로 빠르게 유사 사진을 그룹화합니다.")
                dist_thr = st.slider("유사도 임계(Hamming)", 0, 32, 8, key="sim_thr_simple")
                scope = st.selectbox("대상 범위", ["현재 페이지", "필터된 전체"], index=1, key="sim_scope_simple")
                preview_groups = st.number_input("미리보기 그룹 개수", min_value=1, value=20, step=1, key="sim_prev_simple")

                if st.button("그룹 만들기", key="sim_run_simple"):
                    base = filtered[:48] if scope == "현재 페이지" else filtered
                    items: List[Tuple[str, int]] = []
                    for p in base:
                        img = imread_any_compat(p)
                        if img is None:
                            continue
                        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        items.append((p, phash_from_gray(g)))

                    groups: List[List[str]] = []
                    used = set()
                    for i, (pi, hi) in enumerate(items):
                        if i in used:
                            continue
                        group = [pi]; used.add(i)
                        for j, (pj, hj) in enumerate(items[i+1:], start=i+1):
                            if j in used:
                                continue
                            if hamming_dist(hi, hj) <= dist_thr:
                                group.append(pj); used.add(j)
                        groups.append(group)

                    st.write(f"생성된 그룹: {len(groups)}개")
                    for gi, g in enumerate(groups[:int(preview_groups)], 1):
                        st.markdown(f"**그룹 {gi}** — {len(g)}장")
                        thumbs = []
                        for p in g:
                            th = us.load_thumbnail(p, max_side=256)
                            if th is not None:
                                thumbs.append(cv2.cvtColor(th, cv2.COLOR_BGR2RGB))
                        if thumbs:
                            st.image(thumbs, width=160)

            # 3) RW2 -> JPG 변환
            with st.expander("📷 RW2 → JPG 일괄 변환"):
                st.caption("`rawpy`가 있으면 고품질 변환, 없으면 `imageio` 시도(환경에 따라 실패 가능).")
                out_sub = st.text_input("출력 폴더명", value="_converted", key="raw_out_sub")
                use_rawpy = st.checkbox("rawpy 사용(권장)", value=(rawpy is not None), key="raw_use_rawpy")
                recursive_raw = st.checkbox("하위 폴더 포함(RAW 탐색)", value=recursive, key="raw_recursive")

                if st.button("RW2 변환 실행", key="raw_convert_run"):
                    root_path = Path(root)
                    out_dir = root_path / out_sub
                    out_dir.mkdir(exist_ok=True)
                    if recursive_raw:
                        rw2_list = list(root_path.rglob("*.rw2")) + list(root_path.rglob("*.RW2"))
                    else:
                        rw2_list = list(root_path.glob("*.rw2")) + list(root_path.glob("*.RW2"))

                    if len(rw2_list) == 0:
                        st.warning("RW2 파일을 찾지 못했습니다.")
                    else:
                        done, fail = 0, 0
                        for rp in rw2_list:
                            try:
                                dst = out_dir / (rp.stem + ".jpg")
                                if use_rawpy and rawpy is not None:
                                    with rawpy.imread(str(rp)) as raw:
                                        rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True, output_bps=8)
                                    if iio is None:
                                        from PIL import Image
                                        Image.fromarray(rgb).save(dst, quality=95)
                                    else:
                                        iio.imwrite(dst, rgb, quality=95)
                                else:
                                    if iio is None:
                                        raise RuntimeError("imageio.v3(iio) 가 필요합니다.")
                                    arr = iio.imread(str(rp))
                                    iio.imwrite(dst, arr, quality=95)
                                done += 1
                            except Exception:
                                fail += 1
                        st.success(f"변환 완료: {done}개, 실패: {fail}개 → {out_dir}")

            st.divider()

            # 4) CSV 저장 + 흐린 사진 이동/삭제
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 CSV 저장", use_container_width=True, key="csv_simple"):
                    rows = []
                    for p in paths:
                        r = res.get(p)
                        if not r:
                            continue
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
                if st.button("📁 흐린 사진 이동", use_container_width=True, key="move_blur_simple"):
                    blur_folder = Path(root) / "흐린_사진"
                    blur_folder.mkdir(exist_ok=True)
                    moved = 0
                    for p in paths:
                        r = res.get(p)
                        if not r:
                            continue
                        if r["score"] <= quality_threshold:
                            try:
                                shutil.move(p, blur_folder / Path(p).name)
                                moved += 1
                            except Exception:
                                pass
                    st.success(f"✅ {moved}장을 이동했습니다!")

            with col3:
                if st.button("🗑️ 흐린 사진 삭제", use_container_width=True, key="delete_blur_simple"):
                    deleted = 0
                    for p in paths:
                        r = res.get(p)
                        if not r:
                            continue
                        if r["score"] <= quality_threshold:
                            try:
                                if send2trash:
                                    send2trash(p)
                                else:
                                    os.remove(p)
                                deleted += 1
                            except Exception:
                                pass
                    st.success(f"✅ {deleted}장을 삭제(또는 휴지통 이동)했습니다!")

# ---------------------------------------------------------------------
# 고급 모드
# ---------------------------------------------------------------------
else:
    tab1, tab2, tab3 = st.tabs(["📊 대시보드", "🖼️ 라벨링", "📈 유틸/내보내기"])

    # 대시보드
    with tab1:
        if st.button("🚀 전체 분석", type="primary"):
            paths = us.list_images(root, recursive=recursive)
            st.session_state["paths"] = paths

            params_adv = {
                "long_side": long_side,
                "tiles": tiles,
                "exif_correction": exif_on,
                "face_prior_enabled": face_on,
                "face_prior_alpha": face_alpha,
                "roi_free": roi_free,
                "enable_dl_hybrid": dl_on,
                "dl_weight": dl_weight,
                "dl_motion_bias": dl_motion_bias,
                "dl_weights": (dl_weights_path or None),
            }

            mw = max_workers if isinstance(max_workers, int) else 8

            if use_hybrid:
                res = us.batch_analyze_full_hybrid(paths, params=params_adv, max_workers=mw)
            else:
                res = us.batch_analyze(paths, mode="advanced", tiles=tiles, params=params_adv, max_workers=mw)

            st.session_state["scores"] = res
            st.success(f"✅ {len(res)}개 이미지 분석 완료 (하이브리드={use_hybrid})")

        scores: Dict[str, dict] = st.session_state.get("scores", {})
        if scores:
            rows = [{"path": p,
                     "sharp": sc["sharp_score"],
                     "defocus": sc["defocus_score"],
                     "motion": sc["motion_score"]}
                    for p, sc in scores.items()]
            df_view = pd.DataFrame(rows)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("총 이미지", len(scores))
            with c2:
                st.metric("평균 Sharp", round(df_view["sharp"].mean(), 3))
            with c3:
                st.metric("평균 Motion", round(df_view["motion"].mean(), 3))
            st.dataframe(df_view.head(500), use_container_width=True)

            # 간단 분포 차트(선택)
            st.write("분포(상위 500)")
            st.bar_chart(df_view.head(500)[["sharp", "defocus", "motion"]])

    # 라벨링
    with tab2:
        paths: List[str] = st.session_state.get("paths", [])
        scores: Dict[str, dict] = st.session_state.get("scores", {})
        if not scores:
            st.info("먼저 대시보드에서 전체 분석을 실행하세요.")
        else:
            if "labels" not in st.session_state:
                st.session_state["labels"] = {}

            c1, c2 = st.columns(2)
            with c1:
                per_page = st.selectbox("페이지당 썸네일 수", [12, 24, 48], index=2)
            with c2:
                page = st.number_input("페이지(1부터)", min_value=1, value=1, step=1)

            start = (page - 1) * per_page
            end = min(start + per_page, len(paths))
            page_paths = paths[start:end]

            cols = st.columns(4)

            # 현재 파라미터 구성 + 캐시 키
            params_current = {
                "long_side": long_side,
                "tiles": tiles,
                "exif_correction": exif_on,
                "face_prior_enabled": face_on,
                "face_prior_alpha": face_alpha,
                "roi_free": roi_free,
                "enable_dl_hybrid": dl_on,
                "dl_weight": dl_weight,
                "dl_motion_bias": dl_motion_bias,
                "dl_weights": (dl_weights_path or None),
            }
            params_key = _params_key_for_cache(tiles, params_current)

            for i, p in enumerate(page_paths):
                col = cols[i % 4]
                with col:
                    th = us.load_thumbnail(p, max_side=384)
                    if th is not None:
                        col.image(cv2.cvtColor(th, cv2.COLOR_BGR2RGB), use_container_width=True)

                    sc = scores.get(p)
                    if not sc:
                        sc = compute_scores_cached(p, tiles=tiles, params=params_current, params_key=params_key)
                        scores[p] = sc

                    s, d, m = sc["sharp_score"], sc["defocus_score"], sc["motion_score"]
                    pred = max(("sharp", s), ("defocus", d), ("motion", m), key=lambda x: x[1])[0]
                    current = st.session_state["labels"].get(p, pred)
                    new_label = col.selectbox(
                        label=f"{Path(p).name[:20]}...\nS:{s:.2f} D:{d:.2f} M:{m:.2f}",
                        options=["sharp", "defocus", "motion"],
                        index=["sharp", "defocus", "motion"].index(current),
                        key=make_widget_key("sel_adv", p)
                    )
                    st.session_state["labels"][p] = new_label

                    if col.button("🔎 고해상도", key=make_widget_key("zoom_adv", p)):
                        def _render():
                            big = load_fullres(p, max_side=2048)
                            if big is not None:
                                st.image(big, use_container_width=True)
                        show_modal(f"고해상도 · {Path(p).name}", _render, width="large")

                    if os.name == "nt" and col.button("📂 위치 열기", key=make_widget_key("open_loc", p)):
                        try:
                            os.startfile(str(Path(p).parent))
                        except Exception:
                            st.warning("탐색기를 열 수 없습니다.")

            st.divider()

            # 자동 라벨링(임계값 기반)
            with st.expander("⚡ 자동 라벨링(임계값 기반)"):
                st.caption("현재 스코어 기준으로 라벨을 일괄 부여합니다.")
                min_sharp = st.slider("선명 최소 스코어", 0.0, 1.0, 0.35, 0.01, key="min_sharp_adv")
                min_def = st.slider("아웃포커스 최소 스코어", 0.0, 1.0, 0.35, 0.01, key="min_def_adv")
                min_mot = st.slider("모션 최소 스코어", 0.0, 1.0, 0.35, 0.01, key="min_mot_adv")
                scope = st.selectbox("대상", ["현재 페이지", "전체(분석된 항목)"], index=0, key="auto_scope")

                if st.button("자동 라벨 적용", key="auto_label_run"):
                    base = page_paths if scope == "현재 페이지" else list(scores.keys())
                    applied = 0
                    for p in base:
                        sc = scores.get(p)
                        if not sc:
                            continue
                        sharp_s, def_s, mot_s = sc["sharp_score"], sc["defocus_score"], sc["motion_score"]
                        pred = max([("sharp", sharp_s), ("defocus", def_s), ("motion", mot_s)], key=lambda x: x[1])[0]
                        if pred == "sharp" and sharp_s < min_sharp:
                            pred = "defocus" if def_s >= max(min_def, mot_s) else "motion"
                        if pred == "defocus" and def_s < min_def:
                            pred = "sharp" if sharp_s >= max(min_sharp, mot_s) else "motion"
                        if pred == "motion" and mot_s < min_mot:
                            pred = "sharp" if sharp_s >= max(min_sharp, def_s) else "defocus"
                        st.session_state["labels"][p] = pred
                        applied += 1
                    st.success(f"자동 라벨 적용: {applied}개")

            # 유사도 묶기(near-duplicate)
            with st.expander("🧩 비슷한 사진 묶기(near-duplicate, pHash)"):
                st.caption("pHash + Hamming 거리로 빠른 유사 사진 그룹화.")
                dist_thr = st.slider("유사도 임계(Hamming)", 0, 32, 8, key="sim_thr_adv")
                scope = st.selectbox("대상", ["현재 페이지", "전체(분석된 항목)"], index=1, key="sim_scope_adv")
                preview_groups = st.number_input("미리보기 그룹 개수", min_value=1, value=20, step=1, key="sim_prev_adv")
                best_keep_label = st.selectbox("베스트 1장 라벨", ["sharp", "defocus", "motion"], index=0, key="sim_best_lab")

                if st.button("그룹 생성", key="sim_run_adv"):
                    base = page_paths if scope == "현재 페이지" else list(scores.keys())
                    items: List[Tuple[str, int]] = []
                    for p in base:
                        img = imread_any_compat(p)
                        if img is None:
                            continue
                        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        items.append((p, phash_from_gray(g)))

                    groups: List[List[str]] = []
                    used = set()
                    for i, (pi, hi) in enumerate(items):
                        if i in used:
                            continue
                        group = [pi]; used.add(i)
                        for j, (pj, hj) in enumerate(items[i+1:], start=i+1):
                            if j in used:
                                continue
                            if hamming_dist(hi, hj) <= dist_thr:
                                group.append(pj); used.add(j)
                        if len(group) > 1:
                            groups.append(group)

                    st.session_state["nd_groups"] = groups
                    st.success(f"그룹 수: {len(groups)}")

                groups = st.session_state.get("nd_groups", [])
                show_n = min(len(groups), int(preview_groups))
                for gi, g in enumerate(groups[:show_n], 1):
                    st.markdown(f"**그룹 {gi}** — {len(g)}장")
                    thumbs = []
                    for p in g:
                        th = us.load_thumbnail(p, max_side=256)
                        if th is not None:
                            thumbs.append(cv2.cvtColor(th, cv2.COLOR_BGR2RGB))
                    if thumbs:
                        st.image(thumbs, width=160)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button(f"그룹 라벨=sharp ({gi})"):
                            for p in g:
                                st.session_state["labels"][p] = "sharp"
                    with c2:
                        if st.button(f"그룹 라벨=defocus ({gi})"):
                            for p in g:
                                st.session_state["labels"][p] = "defocus"
                    with c3:
                        if st.button(f"그룹 라벨=motion ({gi})"):
                            for p in g:
                                st.session_state["labels"][p] = "motion"

                    if st.button(f"베스트 1장 {best_keep_label}, 나머지 defocus ({gi})"):
                        best_p = None
                        best_val = -1
                        for p in g:
                            sc = scores.get(p)
                            if not sc:
                                continue
                            if sc["sharp_score"] > best_val:
                                best_val = sc["sharp_score"]; best_p = p
                        if best_p:
                            for p in g:
                                st.session_state["labels"][p] = "defocus"
                            st.session_state["labels"][best_p] = best_keep_label
                            st.success(f"베스트: {Path(best_p).name}")

    # 유틸/내보내기
    with tab3:
        labels: Dict[str, str] = st.session_state.get("labels", {})
        move_or_copy = st.selectbox("내보내기 방식", ["copy", "move"])

        if st.button("📦 학습셋 내보내기"):
            out_root = Path(root) / "train"
            n, outp = us.export_labeled_dataset(labels, out_root, move=(move_or_copy == "move"))
            st.success(f"✅ {n}개 파일 내보내기 → {outp}")

        if st.button("💾 라벨 CSV 저장"):
            if not labels:
                st.warning("라벨이 없습니다.")
            else:
                rows = [{"path": p, "label": lab} for p, lab in labels.items()]
                df = pd.DataFrame(rows)
                out_csv = Path(root) / "labels.csv"
                df.to_csv(out_csv, index=False, encoding="utf-8-sig")
                st.success(f"저장: {out_csv}")

        with st.expander("🏷️ 파일명 자동 태깅/변경(라벨/점수 기반)"):
            st.caption("패턴 예: {stem}_{label} 또는 {stem}_S{sharp} (sharp은 0~1 스코어*100 정수)")
            rule = st.text_input("패턴", value="{stem}_{label}", key="rename_adv_rule")
            dry_run = st.checkbox("미리보기(실제 변경 안 함)", value=True, key="rename_adv_dry")
            scope = st.selectbox("대상", ["전체(라벨된 항목)"], index=0, key="rename_adv_scope")
            if st.button("파일명 적용(고급)", key="rename_adv_apply"):
                scores = st.session_state.get("scores", {})
                base = [p for p in labels.keys()]
                renamed = 0; conflict = 0; failed = 0
                for p in base:
                    lab = labels.get(p, "")
                    sc = scores.get(p, {})
                    sharp = sc.get("sharp_score", 0.0)
                    stem, ext = Path(p).stem, Path(p).suffix
                    new_name = rule.format(stem=stem, label=lab, sharp=int(round(sharp * 100)))
                    dst = Path(p).with_name(new_name + ext)
                    if dst.exists():
                        conflict += 1
                        continue
                    if not dry_run:
                        try:
                            os.rename(p, dst)
                            # 세션 갱신
                            if "scores" in st.session_state and p in st.session_state["scores"]:
                                st.session_state["scores"][str(dst)] = st.session_state["scores"].pop(p)
                            if "labels" in st.session_state and p in st.session_state["labels"]:
                                st.session_state["labels"][str(dst)] = st.session_state["labels"].pop(p)
                            renamed += 1
                        except Exception:
                            failed += 1
                st.success(f"완료: {renamed}, 충돌: {conflict}, 실패: {failed}")

# ---------------------------------------------------------------------
# 푸터
# ---------------------------------------------------------------------
st.divider()
st.caption("💡 팁: 용도에 맞는 모드를 선택하세요 | 간단 모드 = 빠른 검사 | 고급 모드 = 상세 분석")
