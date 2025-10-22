### 📘 README.md

# 📷 Unified Sort — 통합 이미지 품질 검사 및 분류 도구

> Streamlit 기반의 이미지 품질 분석·분류 툴  
> 간단 모드(빠른 선명도 검사)와 고급 모드(다중 특징 분석 + 라벨링 + 학습셋 생성)를 지원합니다.

---

## 🧩 주요 기능

### 🎯 간단 모드
- 빠른 선명도 측정 (`Laplacian`, `Sobel`, `Edge` 기반)
- 선명 / 흐림 자동 구분
- 흐린 사진 자동 이동·삭제 기능
- 점수 CSV 내보내기

### ⚙️ 고급 모드
- 이미지의 다중 저수준 특징 분석
- 3-클래스 분류: `sharp`, `defocus`, `motion`
- 타일링 기반 국소 분석
- 멀티프로세싱 지원
- 라벨링 UI 및 학습셋 내보내기
- CNN 실험 모듈(선택)

### 🧠 추가 기능
- RW2, HEIC 등 RAW 파일 자동 변환 지원
- 고해상도 프리뷰 (`st.dialog` 기반, 자동 폴백)
- 자동/수동 라벨링 및 폴더 분류
- 유사도 기반 이미지 분류(옵션)

---

## 📦 설치 방법

### 1. 가상환경 생성
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # macOS/Linux
```

### 2. 소스 설치 (editable 모드)
```bash
git clone https://github.com/yourname/unified-sort.git
cd unified-sort
pip install -e .
```

### 3. 실행
```bash
streamlit run app/streamlit_app.py
```

---

## 🧰 폴더 구조
```
unified-sort/
│
├─ pyproject.toml          # 프로젝트 설정
├─ requirements.txt        # 필수 패키지 목록
├─ README.md
│
├─ unified_sort/           # 라이브러리 (기능별 모듈화)
│   ├─ __init__.py
│   ├─ io.py
│   ├─ preview.py
│   ├─ features.py
│   ├─ analysis.py
│   ├─ metrics.py
│   ├─ batch.py
│   ├─ export.py
│   ├─ utils.py
│   ├─ types.py
│   └─ models.py
│
└─ app/
    └─ streamlit_app.py    # Streamlit 인터페이스
```

---

## ⚙️ Requirements

| Category | Libraries |
|-----------|------------|
| Core | `numpy`, `pandas`, `opencv-python`, `pillow`, `plotly` |
| Web UI | `streamlit>=1.35` (≥1.36 시 모달 팝업 자동 활성화) |
| Parallel Processing | `multiprocessing`, `tqdm` |
| Optional (RAW) | `pillow-heif`, `rawpy`, `imageio` |
| Optional (DL) | `torch`, `torchvision` |

---

## 🧠 사용 예시
```bash
# 폴더 내 이미지 자동 분석
python -m unified_sort --mode simple --path ./images

# Streamlit GUI 실행
streamlit run app/streamlit_app.py
```

---

## 🧩 개발 팁
- RW2 변환 시 `imageio[ffmpeg]` 또는 `rawpy` 설치 필요
- GPU가 있다면 `torch.cuda.is_available()` 자동 감지
- Streamlit 1.36 이상이면 `st.dialog()` 기반 고해상도 팝업 작동
- 구버전 Streamlit도 자동 폴백되어 문제없이 실행됩니다.

---

## 🪪 License
MIT License  
Copyright © 2025

---