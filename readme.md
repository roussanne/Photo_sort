# 📷 Unified Image Quality Classifier

**Streamlit 기반 통합 이미지 품질 분석 도구**  
하나의 앱으로 간단 모드와 고급 모드를 전환해  
사진의 선명도, 아웃포커스, 모션블러를 자동 판별하고  
유사도 묶기·자동 태깅·RAW 변환까지 수행합니다.

---

## ✨ Features

### 🎯 간단 모드
- Laplacian 기반 빠른 선명도 검사
- 흐린 사진 탐지 및 이동/삭제
- CSV 내보내기
- 파일명 자동 태깅
- 유사도(pHash) 그룹화
- RW2 → JPG 변환 (`rawpy` 또는 `imageio.v3`)
- HEIC/HEIF 지원 (`pillow-heif`)
- 휴지통 삭제 (`send2trash`)

### ⚙️ 고급 모드
- 7가지 저수준 특징(VoL, Tenengrad 등)
- 멀티프로세싱 / 타일 분석
- 자동 라벨링 / 수동 라벨 교정
- pHash 근사중복 탐지
- 학습셋 내보내기(copy/move)
- 라벨 CSV 입출력

---

## 🧩 Requirements

| 타입 | 패키지 | 설명 |
|------|--------|------|
| 필수 | `streamlit>=1.36` | UI 프레임워크 |
| 필수 | `opencv-python>=4.8` | 영상 처리 |
| 필수 | `numpy>=1.24` | 수치 연산 |
| 필수 | `pandas>=2.0` | 데이터 정리 |
| 필수 | `pillow>=9.5` | 이미지 I/O |
| 선택 | `pillow-heif>=0.13` | HEIC/HEIF 로드 |
| 선택 | `rawpy>=0.18` | RAW(RW2) 현상 |
| 선택 | `imageio>=2.31` | RAW 폴백 |
| 선택 | `send2trash>=1.8` | 안전 삭제 |

> Streamlit ≥ 1.36이면 `st.dialog` 모달 UI, 미만이면 자동 인라인 폴백.

---

## ⚙️ Installation

```bash
# 1️⃣ 개발 모드 설치
pip install -e .

# 2️⃣ 필요 패키지 설치
pip install -r requirements.txt
```

---

## 🚀 Run

```bash
streamlit run app/streamlit_app.py
```

---

## 📂 Structure

```
src/unified_sort/
├─ __init__.py
├─ core.py        # 품질 분석/점수 계산
├─ io_utils.py    # 이미지 입출력 (HEIC, RAW 등)
├─ helpers.py     # 유틸 함수 (pHash, 로더, 모달 등)
app/streamlit_app.py
```

On process
클라우드 스토리지 통합  
프로파일 시스템
배치별 통계

3클래스 CNN 학습/추론 기능 분류 on test
EXIF 활용 done
