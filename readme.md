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

### 🧪 하이브리드 옵션 (NEW!)
- EXIF 메타데이터 활용 보정
- 얼굴 검출 기반 가중치
- ROI-free 피사체 무관 분석
- 딥러닝 NR-IQA 융합
- 스레드 안전한 모델 관리

---

## 🆕 Version 0.1.0 개선사항

### 🐛 버그 수정
1. **예외 처리 개선**: `Exception` → `ImportError`, `ModuleNotFoundError` 등 구체적 예외로 변경
2. **딥러닝 모델 에러 처리**: 타입 검증 및 기본값 처리 강화
3. **스레드 안전성**: 싱글톤 패턴에 락(Lock) 추가
4. **순환 임포트 해결**: 직접 임포트 방식으로 개선

### 🔧 개선사항
1. **입력 검증 강화**: 모든 공개 함수에 입력 타입 검증 추가
2. **파일 존재 확인**: I/O 작업 전 파일 존재 여부 체크
3. **에러 메시지 개선**: 구체적이고 도움이 되는 메시지로 변경
4. **타입 힌트 추가**: 모든 함수에 타입 힌트 적용
5. **문서화 강화**: Docstring을 Google 스타일로 통일

### ✨ 새로운 기능
1. **모델 언로드**: `unload_dl_model()` 함수로 메모리 관리
2. **설치 상태 확인**: `check_installation()`, `print_status()` 유틸리티
3. **데이터셋 검증**: `verify_dataset_structure()` 함수
4. **이미지 유사도 비교**: `compare_images_similarity()` 헬퍼

---

## 🧩 Requirements

| 타입 | 패키지 | 버전 | 설명 |
|------|--------|------|------|
| 필수 | `streamlit` | ≥1.36 | UI 프레임워크 |
| 필수 | `opencv-python` | ≥4.8 | 영상 처리 |
| 필수 | `numpy` | ≥1.24 | 수치 연산 |
| 필수 | `pandas` | ≥2.0 | 데이터 정리 |
| 필수 | `pillow` | ≥9.5 | 이미지 I/O |
| 선택 | `pillow-heif` | ≥0.13 | HEIC/HEIF 로드 |
| 선택 | `rawpy` | ≥0.18 | RAW(RW2) 현상 |
| 선택 | `imageio` | ≥2.31 | RAW 폴백 |
| 선택 | `send2trash` | ≥1.8 | 안전 삭제 |
| 선택 | `torch` | ≥2.0 | 딥러닝 (PyTorch) |

> **Note**: Streamlit ≥1.36이면 `st.dialog` 모달 UI 사용, 미만이면 자동 인라인 폴백.

---

## ⚙️ Installation

### 기본 설치
```bash
# 1️⃣ 저장소 클론
git clone https://github.com/yourusername/unified-sort.git
cd unified-sort

# 2️⃣ 개발 모드 설치
pip install -e .

# 3️⃣ 필수 패키지 설치
pip install -r requirements.txt
```

### 선택적 기능 설치
```bash
# HEIC/HEIF 지원 (iPhone 사진)
pip install pillow-heif

# RAW 이미지 처리
pip install rawpy imageio

# 안전 삭제 (휴지통)
pip install send2trash

# 딥러닝 기능 (PyTorch)
pip install torch torchvision
```

### 설치 확인
```python
import unified_sort as us

# 상태 확인
us.print_status()

# 프로그래매틱 확인
status = us.check_installation()
print(status)
```

---

## 🚀 Quick Start

### Streamlit UI 실행
```bash
streamlit run app/streamlit_app.py
```

### Python 스크립트에서 사용

#### 간단한 분석
```python
import unified_sort as us

# 이미지 목록 가져오기
paths = us.list_images("/path/to/photos", recursive=True)

# 간단 분석 (빠름)
results = us.batch_analyze(paths, mode="simple")

# 결과 출력
for path, result in results.items():
    print(f"{path}: {result['score']} - {result['type']}")
```

#### 하이브리드 분석
```python
import unified_sort as us

# 이미지 목록
paths = us.list_images("/path/to/photos")

# 하이브리드 파라미터 설정
params = {
    "long_side": 1024,
    "tiles": 4,
    "exif_correction": True,
    "face_prior_enabled": True,
    "face_prior_alpha": 0.7,
    "enable_dl_hybrid": True,
    "dl_weight": 0.6,
}

# 하이브리드 분석 실행
results = us.batch_analyze_full_hybrid(paths, params=params, max_workers=8)

# 결과 확인
for path, scores in results.items():
    print(f"{path}:")
    print(f"  Sharp: {scores['sharp_score']:.3f}")
    print(f"  Defocus: {scores['defocus_score']:.3f}")
    print(f"  Motion: {scores['motion_score']:.3f}")
```

#### 중복 이미지 찾기
```python
import unified_sort as us
from pathlib import Path

paths = us.list_images("/path/to/photos")

# 이미지 해시 계산
from unified_sort.io_utils import imread_any
import cv2

hashes = {}
for path in paths:
    img = imread_any(path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hashes[path] = us.phash_from_gray(gray)

# 유사 이미지 그룹화
groups = []
used = set()

for i, (path1, hash1) in enumerate(hashes.items()):
    if path1 in used:
        continue
    
    group = [path1]
    used.add(path1)
    
    for path2, hash2 in list(hashes.items())[i+1:]:
        if path2 in used:
            continue
        
        if us.hamming_dist(hash1, hash2) <= 8:
            group.append(path2)
            used.add(path2)
    
    if len(group) > 1:
        groups.append(group)

print(f"Found {len(groups)} duplicate groups")
```

---

## 📂 Project Structure

```
unified-sort/
├── README.md                    # 이 파일
├── LICENSE                      # MIT License
├── requirements.txt             # 의존성 목록
├── pyproject.toml              # 패키지 메타데이터
├── setup.cfg                   # 설정 파일
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
└── src/
    └── unified_sort/
        ├── __init__.py         # 패키지 초기화
        ├── core.py             # 핵심 분석 함수
        ├── io_utils.py         # I/O 유틸리티
        ├── helpers.py          # 헬퍼 함수
        └── pipeline.py         # 하이브리드 파이프라인
```

---

## 🎓 Advanced Usage

### 메모리 관리
```python
import unified_sort as us

# 딥러닝 모델 사용
results = us.batch_analyze_full_hybrid(paths, params={...})

# 메모리 해제 (필요시)
us.unload_dl_model()
```

### 데이터셋 내보내기
```python
import unified_sort as us
from pathlib import Path

# 라벨링 결과 (예시)
labels = {
    "/path/to/img1.jpg": "sharp",
    "/path/to/img2.jpg": "defocus",
    "/path/to/img3.jpg": "motion",
}

# 학습셋으로 내보내기
out_root = Path("/path/to/output")
count, output_path = us.export_labeled_dataset(labels, out_root, move=False)

print(f"Exported {count} images to {output_path}")

# 데이터셋 검증
from unified_sort.io_utils import verify_dataset_structure
stats = verify_dataset_structure(output_path)
print(f"Dataset stats: {stats}")
```

---

## 🐛 Troubleshooting

### 문제: "Core module is not available" 에러
**원인**: 패키지가 제대로 설치되지 않음  
**해결**: 
```bash
pip install -e .
```

### 문제: HEIC 이미지를 읽을 수 없음
**원인**: pillow-heif가 설치되지 않음  
**해결**:
```bash
pip install pillow-heif
```

### 문제: 딥러닝 모델 로딩 실패
**원인**: PyTorch가 설치되지 않았거나 모델 파일이 없음  
**해결**:
```bash
pip install torch torchvision
```

### 문제: 멀티프로세싱 에러
**원인**: Windows에서 `__main__` 가드 없이 실행  
**해결**:
```python
if __name__ == "__main__":
    # 코드 실행
    results = us.batch_analyze(...)
```

---

## 📝 Development Status

### ✅ 완료
- 핵심 분석 엔진
- Streamlit UI (간단/고급 모드)
- EXIF 메타데이터 활용
- 하이브리드 파이프라인
- 스레드 안전성
- 에러 처리 강화

### 🔄 진행 중
- 3클래스 CNN 모델 학습
- 얼굴 검출 모듈 (`detection.py`)
- EXIF 조정 모듈 (`exif_adjust.py`)
- 딥러닝 융합 모듈 (`hybrid.py`)
- NR-IQA 모델 (`nn_iqa.py`)

### 📋 계획
- 클라우드 스토리지 통합 (Google Drive, Dropbox)
- 프로파일 시스템 (설정 저장/로드)
- 배치별 상세 통계
- REST API 제공
- CLI 도구

---

## 🤝 Contributing

버그 리포트, 기능 제안, Pull Request를 환영합니다!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- OpenCV community for excellent image processing tools
- Streamlit team for the amazing UI framework
- PyTorch team for deep learning capabilities

---
