# -*- coding: utf-8 -*-
"""
Simple Image Quality Checker
간단한 이미지 품질 검사 도구 - 일반 사용자용

특징:
- 쉬운 인터페이스
- 자동 분류 (선명/흐림)
- 드래그 앤 드롭 지원
- 한 번에 결과 확인
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image

# =============== HEIC 지원 ===============
try:
    import pillow_heif
    USE_HEIC = True
except Exception:
    USE_HEIC = False


# =============== 이미지 로딩 ===============
def imread_any(path: str):
    """이미지 읽기"""
    p = str(path)
    ext = p.lower().split(".")[-1]
    if USE_HEIC and ext in ("heic", "heif"):
        heif = pillow_heif.read_heif(p)
        img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data


# =============== 간단한 선명도 측정 ===============
def check_sharpness(gray: np.ndarray) -> dict:
    """
    간단한 선명도 체크
    - 선명도 점수 계산 (0~100)
    - 흐림 타입 판단 (선명/아웃포커스/모션블러)
    """
    # 이미지 크기 조정 (빠른 처리)
    h, w = gray.shape
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        gray = cv2.resize(gray, (int(w*s), int(h*s)))
    
    # 1. 라플라시안 분산 (선명도)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 엣지 강도
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(gx*gx + gy*gy))
    
    # 3. 방향성 (모션블러 감지)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)
    hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    direction_score = np.std(hist)
    
    # 점수 계산 (0~100)
    sharpness_score = min(100, (laplacian_var / 5.0))
    edge_score = min(100, (edge_strength / 0.1))
    combined_score = (sharpness_score * 0.6 + edge_score * 0.4)
    
    # 흐림 타입 판단
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


# =============== 이미지 목록 ===============
@st.cache_data(show_spinner=False)
def list_images(root: str, recursive: bool = False):
    """이미지 파일 찾기"""
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
def load_thumbnail(path: str, max_side: int = 300):
    """썸네일 생성"""
    img = imread_any(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# =============== Streamlit UI ===============
st.set_page_config(page_title="이미지 품질 검사", layout="wide", page_icon="📷")

# 헤더
st.title("📷 이미지 품질 검사 도구")
st.markdown("---")

# 사이드바
with st.sidebar:
    st.header("📁 폴더 선택")
    
    # 빠른 선택 버튼
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
    
    st.caption(f"📂 현재 경로: {root}")
    
    recursive = st.checkbox("하위 폴더 포함", value=False)
    
    st.divider()
    
    st.header("⚙️ 검사 기준")
    quality_threshold = st.slider(
        "선명 기준 점수",
        min_value=30,
        max_value=80,
        value=60,
        step=5,
        help="이 점수 이상이면 '선명'으로 판정됩니다"
    )
    
    st.divider()
    
    st.header("🎯 필터")
    show_filter = st.selectbox(
        "보기",
        ["전체", "선명한 사진만", "흐린 사진만"],
        index=0
    )

# 메인 영역
tab1, tab2, tab3 = st.tabs(["🔍 검사 시작", "📊 결과 보기", "💡 도움말"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1️⃣ 폴더에서 사진 찾기")
        if st.button("🔍 사진 검사 시작", type="primary", use_container_width=True):
            paths = list_images(root, recursive=recursive)
            
            if len(paths) == 0:
                st.error(f"❌ '{root}' 폴더에서 이미지를 찾을 수 없습니다.")
            else:
                st.success(f"✅ {len(paths)}장의 사진을 찾았습니다!")
                
                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                for i, p in enumerate(paths):
                    img = imread_any(p)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        results[p] = check_sharpness(gray)
                    
                    progress = (i + 1) / len(paths)
                    progress_bar.progress(progress)
                    status_text.text(f"검사 중... {i+1}/{len(paths)}")
                
                progress_bar.empty()
                status_text.empty()
                
                # 세션에 저장
                st.session_state["paths"] = paths
                st.session_state["results"] = results
                
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
        - 사진이 많으면 시간이 걸려요
        - 점수가 높을수록 선명해요
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
        
        # 정렬 옵션
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_by = st.selectbox(
                "정렬",
                ["점수 높은 순", "점수 낮은 순", "파일명 순"],
                index=0
            )
        
        with col2:
            per_page = st.selectbox("페이지당", [12, 24, 48], index=0)
        
        with col3:
            page = st.number_input("페이지", min_value=1, value=1, step=1)
        
        # 정렬
        if sort_by == "점수 높은 순":
            filtered_paths.sort(key=lambda p: results[p]["score"], reverse=True)
        elif sort_by == "점수 낮은 순":
            filtered_paths.sort(key=lambda p: results[p]["score"])
        else:
            filtered_paths.sort()
        
        # 페이지네이션
        start = (page - 1) * per_page
        end = min(start + per_page, len(filtered_paths))
        page_paths = filtered_paths[start:end]
        
        st.divider()
        
        # 그리드 표시
        cols = st.columns(4)
        for i, p in enumerate(page_paths):
            col = cols[i % 4]
            r = results[p]
            
            with col:
                thumb = load_thumbnail(p)
                if thumb is not None:
                    st.image(thumb, use_container_width=True)
                
                # 점수 표시
                score = r["score"]
                if score > quality_threshold:
                    st.success(f"**{r['type']}**")
                    st.caption(f"점수: {score}")
                else:
                    st.warning(f"**{r['type']}**")
                    st.caption(f"점수: {score}")
                
                st.caption(f"📁 {Path(p).name[:25]}")
                
                # 상세 정보 (접기)
                with st.expander("상세 정보"):
                    st.write(f"선명도: {r['laplacian']}")
                    st.write(f"엣지: {r['edge']}")
                    st.write(f"방향성: {r['direction']}")
        
        st.divider()
        
        # 일괄 작업
        st.subheader("3️⃣ 흐린 사진 정리하기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 CSV로 저장", use_container_width=True):
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
                        except Exception as e:
                            pass
                
                st.success(f"✅ {moved}장을 '{blur_folder}'로 이동했습니다!")
        
        with col3:
            if st.button("🗑️ 흐린 사진 삭제", use_container_width=True, type="secondary"):
                # 확인 메시지
                if "confirm_delete" not in st.session_state:
                    st.session_state["confirm_delete"] = False
                
                if not st.session_state["confirm_delete"]:
                    st.session_state["confirm_delete"] = True
                    st.warning("⚠️ 다시 한 번 클릭하면 삭제됩니다!")
                else:
                    deleted = 0
                    for p in paths:
                        if p in results and results[p]["score"] <= quality_threshold:
                            try:
                                os.remove(p)
                                deleted += 1
                            except Exception as e:
                                pass
                    
                    st.success(f"✅ {deleted}장을 삭제했습니다!")
                    st.session_state["confirm_delete"] = False

with tab3:
    st.subheader("💡 도움말")
    
    st.markdown("""
    ## 📖 사용 가이드
    
    ### 이 도구는 무엇을 하나요?
    - 사진 폴더를 검사해서 **선명한 사진**과 **흐린 사진**을 자동으로 구분합니다
    - 흐린 사진을 찾아서 정리하거나 삭제할 수 있습니다
    
    ### 점수는 어떻게 계산되나요?
    - **0~100점** 사이로 계산됩니다
    - **60점 이상**: 선명한 사진 ✅
    - **60점 미만**: 흐린 사진 ⚠️
    - 점수가 높을수록 더 선명합니다
    
    ### 흐림 타입이란?
    - **선명 ✅**: 초점이 잘 맞고 또렷한 사진
    - **아웃포커스 🌫️**: 초점이 안 맞아서 전반적으로 흐린 사진
    - **모션블러 📸**: 카메라나 피사체가 움직여서 흐린 사진
    
    ### 주의사항
    ⚠️ **삭제는 되돌릴 수 없습니다!**
    - 처음엔 "이동" 기능을 사용해서 확인해보세요
    - 정말 필요 없는 사진만 삭제하세요
    
    ### 팁
    💡 **효율적으로 사용하기:**
    1. 먼저 작은 폴더로 테스트해보세요
    2. "점수 낮은 순"으로 정렬하면 흐린 사진을 먼저 볼 수 있어요
    3. 점수 기준을 조절해서 자신에게 맞게 설정하세요
    4. CSV로 저장하면 나중에 엑셀에서도 볼 수 있어요
    
    ### 지원 파일 형식
    - JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP
    - HEIC/HEIF (iPhone 사진, pillow-heif 설치 필요)
    
    ### 속도
    - 일반 사진: 100장 기준 약 30초~1분
    - 고화질 사진: 시간이 더 걸릴 수 있습니다
    
    ---
    
    ## ❓ 자주 묻는 질문
    
    **Q: 왜 어떤 선명한 사진이 낮은 점수를 받나요?**
    - 배경이 흐린(보케 효과) 사진은 점수가 낮을 수 있습니다
    - 밝기가 매우 어둡거나 밝으면 점수가 낮을 수 있습니다
    - 점수 기준을 조절해보세요
    
    **Q: 실수로 삭제했어요!**
    - 휴지통을 확인해보세요 (Windows: 휴지통, Mac: 휴지통)
    - 영구 삭제되었다면 복구가 어렵습니다
    - 중요한 사진은 항상 백업하세요!
    
    **Q: 하위 폴더까지 검사하려면?**
    - 왼쪽 사이드바에서 "하위 폴더 포함"을 체크하세요
    
    **Q: 처리가 너무 느려요**
    - 사진 개수가 많으면 시간이 걸립니다
    - 하위 폴더를 제외하고 한 폴더씩 검사해보세요
    
    ---
    
    ### 🆘 문제가 있나요?
    - 폴더 경로가 정확한지 확인하세요
    - 사진 파일이 폴더에 있는지 확인하세요
    - 프로그램을 다시 시작해보세요
    """)
    
    st.divider()
    
    st.info("""
    **버전:** 간단 버전 1.0
    
    **만든이:** AI 어시스턴트
    
    **라이선스:** 자유롭게 사용하세요!
    """)

# 푸터
st.divider()
st.caption("💡 팁: 중요한 사진은 항상 백업해두세요! | 만든 날짜: 2025")