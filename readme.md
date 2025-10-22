# 📷 Unified Photo Sort (통합 이미지 품질 검사)

`Unified Photo Sort`는 사진 폴더를 한 번에 훑어보고 선명도 문제를 찾아내는 Streamlit 기반 데스크탑 앱입니다. 기본 제공하는 **간단 모드**와 **고급 모드**를 오가며 빠르게 결과를 확인하거나, 세부 지표와 라벨을 조정해 학습 데이터셋까지 만들어 낼 수 있습니다. 내부 로직은 `unified_sort` 파이썬 패키지로 분리되어 있어 별도 스크립트에서도 재사용할 수 있습니다.

---

## ✨ 주요 기능
- **간단 모드** – 폴더를 선택하고 "검사 시작"만 누르면 선명도 점수(0~100)와 판정 결과를 즉시 확인
- **고급 모드** – 타일 기반 특징량(VoL, Tenengrad, Edge Spread 등)으로 Sharp/Defocus/Motion 점수를 동시에 산출
- **썸네일 & 고해상도 미리보기** – 최대 48장의 썸네일을 페이지 없이 스크롤하며 확인, 필요 시 모달로 원본 확대
- **수동 라벨 편집 및 내보내기** – 예측 결과를 드롭다운으로 수정하고 `train/{sharp,defocus,motion}` 구조로 복사/이동
- **대용량 폴더 대응** – 멀티 스레드 처리, 하위 폴더 재귀 검색, 긴 변 리사이즈 조절 등 속도/품질 튜닝 옵션 제공
- **HEIC / RAW(RW2) 선택 지원** – `pillow-heif`, `rawpy` 설치 시 아이폰 HEIC와 파나소닉 RAW도 바로 불러오기

---

## ⚙️ 환경 요구 사항
- Python 3.9 이상 (3.10/3.11 권장)
- 필수 패키지: `pip install -r requirements.txt`
  - `streamlit`, `numpy`, `pandas`, `opencv-python`, `Pillow`, `plotly`
- 선택 패키지 (필요 시 개별 설치)
  - HEIC/HEIF: `pip install pillow-heif`
  - Panasonic RW2 등 RAW: `pip install rawpy imageio[v3]`

---

## 🚀 빠른 시작
1. 저장소를 클론하거나 소스 코드를 다운로드합니다.
2. (선택) 가상환경을 활성화합니다.
3. 필수 의존성을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
4. Streamlit 앱을 실행합니다.
   ```bash
   streamlit run unified-sort/app/streamlit_app.py
   ```
5. 브라우저에서 `http://localhost:8501`에 접속하면 UI가 열립니다.

> **Tip:** Windows에서 네트워크 드라이브/한글 경로를 사용할 때는 "직접 입력" 옵션으로 전체 경로를 붙여 넣으면 안정적으로 인식됩니다.

---

## 🧭 앱 워크플로
### 🎯 간단 모드
1. 사이드바에서 폴더를 선택하고 "🔍 검사 시작"을 누릅니다.
2. 모든 이미지에 대해 선명도 점수와 간단한 품질 판정(선명/모션/아웃포커스)을 계산합니다.
3. "보기" 필터로 선명/흐림 이미지를 나눠 확인하고, 필요하면 고해상도 미리보기를 엽니다.

### ⚙️ 고급 모드
1. "🚀 전체 분석"으로 폴더 전체를 분석하면 Sharp/Defocus/Motion 세 점수가 저장됩니다.
2. "📊 대시보드" 탭에서 최근 200개의 스코어를 데이터프레임으로 검토합니다.
3. "🖼️ 라벨링" 탭에서 각 이미지의 예측 라벨을 확인하고 드롭다운으로 직접 수정합니다.
4. "📦 학습셋 내보내기" 버튼으로 `train/` 하위 폴더에 복사(copy) 또는 이동(move)할 수 있습니다.

---

## 🧪 파이썬 API 사용 예시
```python
from unified_sort import list_images, batch_analyze

paths = list_images("/path/to/photos", recursive=True)
results = batch_analyze(paths, mode="advanced", tiles=4, params={"long_side": 1024})

sharp_scores = {p: r["sharp_score"] for p, r in results.items()}
```
- `mode="simple"`을 사용하면 `score`, `type`, `quality`가 담긴 간단 선명도 결과가 반환됩니다.
- `mode="advanced"`는 `sharp_score`, `defocus_score`, `motion_score` 및 사용된 특징량을 포함합니다.

---

## 🗂️ 프로젝트 구조
```
Photo_sort/
├─ unified-sort/app/streamlit_app.py   # 메인 Streamlit 애플리케이션
├─ unified-sort/src/unified_sort/      # 재사용 가능한 분석/입출력 로직
├─ requirements.txt
└─ readme.md
```

---

## ❓ 문제 해결 가이드
- **`ModuleNotFoundError: pillow_heif`** → HEIC 파일을 열어야 한다면 `pip install pillow-heif` 후 다시 실행합니다.
- **RAW(RW2) 파일이 읽히지 않을 때** → `pip install rawpy imageio[v3]` 설치 후 앱을 재시작합니다.
- **OpenCV가 이미지 로드에 실패** → 권한 또는 경로(특히 네트워크 드라이브) 문제일 수 있습니다. `cv2.imdecode`는 너무 긴 경로에 취약하므로 폴더를 로컬로 복사해 확인해 보세요.
- **분석이 느릴 때** → 긴 변 리사이즈 값을 낮추거나(예: 1024 → 896), 타일 수를 줄이고, 워커 수를 4~8 사이로 조정해 보세요.

---

## 📄 라이선스
현재 저장소에는 별도의 라이선스가 포함되어 있지 않습니다. 배포 전 프로젝트 정책에 맞는 라이선스를 추가해 주세요.
