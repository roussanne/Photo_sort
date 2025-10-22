# -*- coding: utf-8 -*-
from pathlib import Path
import streamlit as st
import pandas as pd
import cv2
import unified_sort as us

st.set_page_config(page_title="통합 이미지 품질 검사", layout="wide", page_icon="📷")
st.title("📷 이미지 품질 검사 도구")

# --- modal compatibility helper (st.dialog decorator or inline fallback) ---
def show_modal(title: str, render_fn, width: str = "large"):
    """
    Streamlit 1.36+ : st.dialog(title, width=...) 데코레이터 기반 모달
    이전 버전        : 인라인 컨테이너로 폴백
    사용법:
        def _render():
            st.image(...)
        show_modal("제목", _render, width="large")
    """
    dlg = getattr(st, "dialog", None)
    if callable(dlg):
        @dlg(title, width=width)   # 데코레이터로 모달 정의
        def _inner():
            render_fn()
        _inner()  # 즉시 실행
    else:
        st.subheader(title)        # 인라인 폴백
        render_fn()
        st.markdown("---")

mode_col1, mode_col2 = st.columns([3, 1])
with mode_col1:
    st.markdown("하나의 도구로 간단한 검사부터 전문적인 분석까지 모두 가능합니다")
with mode_col2:
    app_mode = st.selectbox("사용 모드", ["🎯 간단 모드", "⚙️ 고급 모드"])
is_simple = (app_mode == "🎯 간단 모드")
st.markdown("---")

# 사이드바
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
            long_side = st.slider("분석 리사이즈(긴 변)", 640, 2048, 1024, 64)
            tiles = st.slider("타일 수 (NxN)", 2, 6, 4, 1)
            max_workers = st.slider("워커 수", 1, 16, 8)
        with st.expander("🎚️ 가중치(선택)"):
            w = {}
            w["w_sharp_vol"] = st.slider("VoL", 0.0, 1.0, 0.30, 0.01)
            w["w_sharp_ten"] = st.slider("Tenengrad", 0.0, 1.0, 0.25, 0.01)
            w["w_sharp_hfr"] = st.slider("HighFreqRatio", 0.0, 1.0, 0.20, 0.01)
            w["w_sharp_esw"] = st.slider("EdgeSpread(역)", 0.0, 1.0, 0.15, 0.01)
            w["w_sharp_slope"] = st.slider("RadialSlope(역)", 0.0, 1.0, 0.10, 0.01)
            w["w_def_esw"] = st.slider("Defocus: EdgeSpread", 0.0, 1.0, 0.40, 0.01)
            w["w_def_vol"] = st.slider("Defocus: VoL(역)", 0.0, 1.0, 0.25, 0.01)
            w["w_def_slope"] = st.slider("Defocus: RadialSlope(역)", 0.0, 1.0, 0.25, 0.01)
            w["w_def_aniso"] = st.slider("Defocus: Anisotropy(역)", 0.0, 1.0, 0.10, 0.01)
            w["w_mot_aniso"] = st.slider("Motion: Anisotropy", 0.0, 1.0, 0.60, 0.01)
            w["w_mot_strat"] = st.slider("Motion: StructureTensor", 0.0, 1.0, 0.30, 0.01)
            w["w_mot_volinv"] = st.slider("Motion: VoL(역)", 0.0, 1.0, 0.10, 0.01)

params = dict(long_side=1024) if is_simple else dict(long_side=long_side, **w)

# ============== 간단 모드 ==============
if is_simple:
    tab1, tab2 = st.tabs(["🔍 검사 시작", "📊 결과 보기"])
    with tab1:
        if st.button("🔍 검사 시작", type="primary", use_container_width=True):
            paths = us.list_images(root, recursive=recursive)
            st.session_state["paths"] = paths
            if not paths:
                st.error("이미지를 찾을 수 없습니다.")
            else:
                res = us.batch_analyze(paths, mode="simple", params=params)
                st.session_state["results_simple"] = res
                st.success(f"✅ {len(res)}장 분석 완료")

    with tab2:
        res = st.session_state.get("results_simple", {})
        paths = st.session_state.get("paths", [])
        if not res:
            st.info("먼저 검사 시작을 눌러주세요")
        else:
            filtered = []
            for p in paths:
                if p not in res:
                    continue
                r = res[p]
                score = r["score"] if isinstance(r, dict) else r.score
                if show_filter == "선명한 사진만" and score <= quality_threshold:
                    continue
                if show_filter == "흐린 사진만" and score > quality_threshold:
                    continue
                filtered.append(p)

            st.write(f"표시 중: {len(filtered)}장")
            cols = st.columns(4)
            for i, p in enumerate(filtered[:48]):
                col = cols[i % 4]
                with col:
                    thumb = us.load_thumbnail(p, max_side=320)
                    if thumb is not None:
                        col.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(Path(p).name[:30])

                    if col.button("🔎 고해상도", key=us.make_widget_key("zoom_simple", p)):
                        def _render():
                            big = us.load_fullres(p, max_side=2048)
                            if big is not None:
                                st.image(cv2.cvtColor(big, cv2.COLOR_BGR2RGB), use_container_width=True)
                        show_modal(f"고해상도 · {Path(p).name}", _render, width="large")

# ============== 고급 모드 ==============
else:
    tab1, tab2 = st.tabs(["📊 대시보드", "🖼️ 라벨링"])
    with tab1:
        if st.button("🚀 전체 분석", type="primary"):
            paths = us.list_images(root, recursive=recursive)
            st.session_state["paths"] = paths
            res = us.batch_analyze(paths, mode="advanced", tiles=tiles, params=params, max_workers=max_workers)
            st.session_state["scores"] = res
            st.success(f"✅ {len(res)}개 이미지 분석 완료")
        scores = st.session_state.get("scores", {})
        if scores:
            rows = []
            for p, sc in scores.items():
                rows.append({"path": p, "sharp": sc["sharp_score"], "defocus": sc["defocus_score"], "motion": sc["motion_score"]})
            st.dataframe(pd.DataFrame(rows).head(200), use_container_width=True)

    with tab2:
        paths = st.session_state.get("paths", [])
        scores = st.session_state.get("scores", {})
        if not scores:
            st.info("먼저 대시보드에서 전체 분석을 실행하세요.")
        else:
            if "labels" not in st.session_state:
                st.session_state["labels"] = {}
            cols = st.columns(4)
            for i, p in enumerate(paths[:48]):
                col = cols[i % 4]
                with col:
                    thumb = us.load_thumbnail(p, max_side=384)
                    if thumb is not None:
                        col.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), use_container_width=True)
                    sc = scores.get(p)
                    if not sc:
                        continue
                    s, d, m = sc["sharp_score"], sc["defocus_score"], sc["motion_score"]
                    pred = max(("sharp", s), ("defocus", d), ("motion", m), key=lambda x: x[1])[0]
                    current = st.session_state["labels"].get(p, pred)
                    new_label = col.selectbox(
                        label=f"{Path(p).name[:20]}...\nS:{s:.2f} D:{d:.2f} M:{m:.2f}",
                        options=["sharp","defocus","motion"],
                        index=["sharp","defocus","motion"].index(current),
                        key=us.make_widget_key("sel", p)
                    )
                    st.session_state["labels"][p] = new_label

                    if col.button("🔎 고해상도", key=us.make_widget_key("zoom_adv", p)):
                        def _render():
                            big = us.load_fullres(p, max_side=2048)
                            if big is not None:
                                st.image(cv2.cvtColor(big, cv2.COLOR_BGR2RGB), use_container_width=True)
                        show_modal(f"고해상도 · {Path(p).name}", _render, width="large")

            st.divider()
            move_or_copy = st.selectbox("내보내기 방식", ["copy","move"])
            if st.button("📦 학습셋 내보내기"):
                out_root = Path(root) / "train"
                n, outp = us.export_labeled_dataset(st.session_state["labels"], out_root, move=(move_or_copy=="move"))
                st.success(f"✅ {n}개 파일 내보내기 → {outp}")
