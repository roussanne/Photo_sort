"""
Core analysis functions for image quality assessment.

개선사항:
1. 타입 힌트 추가로 명확성 향상
2. 에러 처리 강화
3. 입력 검증 추가
4. 문서화 개선
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import cv2


def list_images(root: str, recursive: bool = False) -> List[str]:
    """
    지정된 폴더에서 이미지 파일을 찾아 경로 리스트를 반환합니다.
    
    지원하는 형식: JPG, JPEG, PNG, BMP, TIF, TIFF, WEBP (대소문자 무관)
    
    Args:
        root: 검색할 루트 폴더 경로
        recursive: True면 하위 폴더까지 재귀적으로 검색
    
    Returns:
        발견된 이미지 파일의 절대 경로 리스트 (정렬됨)
        폴더가 존재하지 않으면 빈 리스트 반환
    """
    root_path = Path(root)
    
    # 폴더가 존재하지 않거나 유효하지 않은 경로인 경우
    if not root_path.exists():
        return []
    
    if not root_path.is_dir():
        return []
    
    # 지원하는 이미지 확장자 패턴 (대소문자 모두)
    patterns = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP"
    ]
    
    paths = []
    
    try:
        if recursive:
            # 재귀적 검색 (모든 하위 폴더 포함)
            for pattern in patterns:
                paths.extend(root_path.rglob(pattern))
        else:
            # 현재 폴더만 검색
            for pattern in patterns:
                paths.extend(root_path.glob(pattern))
    except (OSError, PermissionError) as e:
        # 접근 권한이 없거나 파일 시스템 오류
        print(f"Warning: Error scanning directory {root}: {e}")
        return []
    
    # Path 객체를 문자열로 변환하고, 파일만 필터링한 후 정렬
    valid_paths = [str(p) for p in paths if p.is_file()]
    
    # 중복 제거 후 정렬 (대소문자가 다른 같은 파일 처리)
    return sorted(set(valid_paths))


def load_thumbnail(path: str, max_side: int = 384) -> Optional[np.ndarray]:
    """
    이미지를 썸네일 크기로 로드합니다.
    
    원본 이미지의 비율을 유지하면서 긴 변이 max_side를 넘지 않도록
    리사이즈합니다. BGR 형식의 numpy 배열로 반환합니다.
    
    Args:
        path: 이미지 파일 경로
        max_side: 썸네일의 최대 변 길이 (픽셀)
    
    Returns:
        BGR 형식의 numpy 배열 (uint8), 실패 시 None
    """
    from .io_utils import imread_any
    
    if not isinstance(max_side, int) or max_side <= 0:
        max_side = 384  # 기본값으로 복구
    
    try:
        img = imread_any(path)
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return None
    
    if img is None:
        return None
    
    # 이미지가 유효한 numpy 배열인지 확인
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
    
    try:
        h, w = img.shape[:2]
        
        # 이미 충분히 작으면 리사이즈 불필요
        if max(h, w) <= max_side:
            return img
        
        # 비율을 유지하면서 축소
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 최소 크기 보장 (1픽셀 이상)
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        img_resized = cv2.resize(
            img, 
            (new_w, new_h), 
            interpolation=cv2.INTER_AREA  # 축소 시 가장 좋은 품질
        )
        
        return img_resized
        
    except Exception as e:
        print(f"Warning: Failed to resize thumbnail for {path}: {e}")
        return None


def _compute_vol(gray: np.ndarray) -> float:
    """
    Variance of Laplacian (VoL) 계산.

    가장 널리 사용되는 블러 감지 메트릭.
    라플라시안 필터의 분산을 계산하여 엣지 강도를 측정합니다.
    높을수록 선명함.
    """
    try:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    except Exception:
        return 0.0


def _compute_tenengrad(gray: np.ndarray, ksize: int = 3) -> float:
    """
    Tenengrad 메트릭 계산.

    Sobel 연산자를 사용한 그래디언트 크기의 제곱 합.
    엣지의 강도와 밀도를 동시에 고려합니다.
    높을수록 선명함.
    """
    try:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(gx**2 + gy**2)
        tenengrad = float(np.mean(magnitude**2))
        return tenengrad
    except Exception:
        return 0.0


def _compute_hfr(gray: np.ndarray, threshold_percentile: float = 90) -> float:
    """
    High Frequency Ratio (HFR) 계산.

    FFT를 사용하여 주파수 도메인에서 고주파 성분의 비율을 계산.
    선명한 이미지는 고주파 성분이 많고, 블러된 이미지는 저주파가 지배적.
    높을수록 선명함.
    """
    try:
        h, w = gray.shape

        # FFT 변환
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # 중심에서의 거리 계산 (주파수)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)

        # 고주파 영역 정의 (중심에서 먼 영역)
        max_distance = np.sqrt(cx**2 + cy**2)
        threshold_distance = max_distance * 0.3  # 외곽 70% 영역

        high_freq_mask = distance > threshold_distance
        low_freq_mask = ~high_freq_mask

        # 고주파 에너지 비율
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
        total_energy = np.sum(magnitude_spectrum) + 1e-8

        hfr = float(high_freq_energy / total_energy)
        return hfr
    except Exception:
        return 0.0


def _compute_esw(gray: np.ndarray, sample_size: int = 100) -> float:
    """
    Edge Spread Width (ESW) 계산.

    엣지의 확산 폭을 측정하여 블러 정도를 평가.
    선명한 이미지는 엣지가 날카롭고(좁은 폭),
    블러된 이미지는 엣지가 넓게 퍼집니다.
    낮을수록 선명함 (역수 취하여 반환).
    """
    try:
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150)

        # 엣지 픽셀 찾기
        edge_coords = np.argwhere(edges > 0)

        if len(edge_coords) < 10:
            return 0.0

        # 샘플링 (너무 많으면 느림)
        if len(edge_coords) > sample_size:
            indices = np.random.choice(len(edge_coords), sample_size, replace=False)
            edge_coords = edge_coords[indices]

        # 각 엣지 포인트에서 그래디언트 프로파일 분석
        widths = []
        for y, x in edge_coords:
            # 3x3 윈도우에서 그래디언트 계산
            y1, y2 = max(0, y-1), min(gray.shape[0], y+2)
            x1, x2 = max(0, x-1), min(gray.shape[1], x+2)

            window = gray[y1:y2, x1:x2]
            if window.size < 4:
                continue

            # 그래디언트 크기
            gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)

            # 전체 그래디언트 중 절반 이상인 영역의 폭
            if magnitude.max() > 0:
                threshold = magnitude.max() * 0.5
                width = np.sum(magnitude > threshold)
                widths.append(width)

        if not widths:
            return 0.0

        # 평균 엣지 폭 (역수로 변환: 좁을수록 선명함)
        avg_width = np.mean(widths)
        # 정규화: 1-9 픽셀 폭을 0-1 범위로 변환
        esw_score = 1.0 / (1.0 + avg_width / 3.0)

        return float(esw_score)
    except Exception:
        return 0.0


def _compute_rss(gray: np.ndarray, tiles: int = 4) -> float:
    """
    Ratio of Sharpness Scores (RSS) 계산.

    이미지를 타일로 나누어 각 타일의 선명도를 계산한 후,
    타일 간 선명도 편차를 분석.
    균일하게 흐린 이미지 vs 부분적으로 흐린 이미지를 구분.
    높을수록 전체적으로 균일하게 선명함.
    """
    try:
        h, w = gray.shape
        tile_h, tile_w = h // tiles, w // tiles

        if tile_h < 10 or tile_w < 10:
            tiles = 2
            tile_h, tile_w = h // tiles, w // tiles

        sharpness_scores = []

        for i in range(tiles):
            for j in range(tiles):
                y1 = i * tile_h
                y2 = (i + 1) * tile_h if i < tiles - 1 else h
                x1 = j * tile_w
                x2 = (j + 1) * tile_w if j < tiles - 1 else w

                tile = gray[y1:y2, x1:x2]

                if tile.size < 100:
                    continue

                # 각 타일의 Laplacian 분산 계산
                lap_var = cv2.Laplacian(tile, cv2.CV_64F).var()
                sharpness_scores.append(lap_var)

        if len(sharpness_scores) < 2:
            return 0.5

        # 평균 대비 표준편차 비율 (변동계수)
        mean_sharpness = np.mean(sharpness_scores)
        std_sharpness = np.std(sharpness_scores)

        if mean_sharpness < 1e-6:
            return 0.0

        # CV (Coefficient of Variation) - 낮을수록 균일함
        cv = std_sharpness / mean_sharpness

        # RSS: 평균 선명도가 높고 CV가 낮을수록 좋음
        # 정규화된 점수
        rss = mean_sharpness / (1.0 + cv * 10)

        return float(rss)
    except Exception:
        return 0.0


def _compute_ai(gray: np.ndarray, num_directions: int = 18) -> float:
    """
    Anisotropy Index (AI) 계산.

    방향별 엣지 에너지 분포를 분석하여 이방성(비등방성)을 측정.
    모션 블러는 특정 방향으로 에너지가 집중되고 (높은 AI),
    디포커스 블러는 모든 방향이 균일함 (낮은 AI).
    높을수록 방향성 블러 (모션블러 가능성).
    """
    try:
        # Sobel로 그래디언트 계산
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2) + 1e-8
        angle = np.arctan2(gy, gx) + np.pi  # 0 ~ 2π

        # 방향별 히스토그램
        hist, _ = np.histogram(
            angle,
            bins=num_directions,
            range=(0, 2 * np.pi),
            weights=magnitude
        )

        # 정규화
        hist = hist / (hist.sum() + 1e-8)

        # 표준편차로 이방성 측정 (높을수록 특정 방향 집중)
        ai = float(np.std(hist))

        return ai
    except Exception:
        return 0.0


def _compute_str(gray: np.ndarray) -> float:
    """
    Spectral Total Variance (STR) 계산.

    주파수 스펙트럼의 전체 분산을 측정.
    선명한 이미지는 다양한 주파수 성분을 가지고 (높은 분산),
    블러된 이미지는 저주파에 집중됨 (낮은 분산).
    높을수록 선명함.
    """
    try:
        # FFT 변환
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # 로그 스케일 변환 (동적 범위 압축)
        log_spectrum = np.log1p(magnitude_spectrum)

        # 분산 계산
        variance = float(np.var(log_spectrum))

        return variance
    except Exception:
        return 0.0


def compute_scores_advanced(
    gray: np.ndarray,
    tiles: int,
    params: dict
) -> Dict[str, float]:
    """
    고급 영상처리 기법으로 이미지 품질 점수를 계산합니다.

    7가지 특징을 추출하여 종합적으로 분석:
    1. VoL (Variance of Laplacian) - 기본 블러 메트릭
    2. Tenengrad - 그래디언트 기반 선명도
    3. HFR (High Frequency Ratio) - 고주파 성분 비율
    4. ESW (Edge Spread Width) - 엣지 확산 폭
    5. RSS (Ratio of Sharpness Scores) - 영역별 선명도 균일성
    6. AI (Anisotropy Index) - 방향성 블러 감지
    7. STR (Spectral Total Variance) - 주파수 스펙트럼 분산

    Args:
        gray: 그레이스케일 이미지 (numpy 배열)
        tiles: 타일 분할 개수 (RSS에서 사용)
        params: 분석 파라미터 딕셔너리 (미래 확장용)

    Returns:
        세 가지 점수를 포함하는 딕셔너리
        {sharp_score: float, defocus_score: float, motion_score: float}
    """
    # 입력 검증
    if not isinstance(gray, np.ndarray):
        return {
            "sharp_score": 0.5,
            "defocus_score": 0.25,
            "motion_score": 0.25
        }

    if gray.size == 0:
        return {
            "sharp_score": 0.5,
            "defocus_score": 0.25,
            "motion_score": 0.25
        }

    # 7가지 특징 추출
    vol = _compute_vol(gray)
    tenengrad = _compute_tenengrad(gray)
    hfr = _compute_hfr(gray)
    esw = _compute_esw(gray)
    rss = _compute_rss(gray, tiles=tiles)
    ai = _compute_ai(gray)
    str_val = _compute_str(gray)

    # 특징 정규화 (0-1 범위로)
    # 이 값들은 실험적으로 결정된 정규화 상수입니다
    vol_norm = min(1.0, vol / 500.0)
    tenengrad_norm = min(1.0, tenengrad / 1000.0)
    hfr_norm = hfr  # 이미 0-1 범위
    esw_norm = esw  # 이미 0-1 범위
    rss_norm = min(1.0, rss / 100.0)
    ai_norm = min(1.0, ai / 0.15)
    str_norm = min(1.0, str_val / 10.0)

    # === 분류 로직 ===

    # 1. 전체적인 선명도 (VoL, Tenengrad, HFR, STR 기반)
    overall_sharpness = (vol_norm * 0.3 +
                         tenengrad_norm * 0.3 +
                         hfr_norm * 0.2 +
                         str_norm * 0.2)

    # 2. 방향성 블러 감지 (AI가 높으면 모션블러)
    directionality = ai_norm

    # 3. 공간적 균일성 (RSS, ESW 기반)
    spatial_uniformity = (rss_norm * 0.6 + esw_norm * 0.4)

    # === 3-클래스 분류 ===

    # Sharp: 전체적으로 선명하고, 균일함
    if overall_sharpness > 0.6 and spatial_uniformity > 0.5:
        sharp_score = 0.7 + overall_sharpness * 0.3
        defocus_score = max(0.0, 0.3 - overall_sharpness * 0.3)
        motion_score = max(0.0, directionality * 0.2)

    # Motion Blur: 방향성이 강하고, 전체적으로 흐림
    elif directionality > 0.6 and overall_sharpness < 0.5:
        motion_score = 0.6 + directionality * 0.3
        sharp_score = max(0.0, overall_sharpness * 0.3)
        defocus_score = max(0.0, 0.4 - directionality * 0.2)

    # Defocus Blur: 방향성 없이 균일하게 흐림
    elif overall_sharpness < 0.5 and directionality < 0.5:
        defocus_score = 0.6 + (1.0 - overall_sharpness) * 0.3
        sharp_score = max(0.0, overall_sharpness * 0.4)
        motion_score = max(0.0, directionality * 0.2)

    # Ambiguous cases: 점수 기반 분배
    else:
        # 선형 조합
        sharp_score = overall_sharpness * 0.5 + spatial_uniformity * 0.3
        motion_score = directionality * 0.5
        defocus_score = (1.0 - overall_sharpness) * 0.4

    # 정규화 (합이 1이 되도록)
    total = sharp_score + defocus_score + motion_score + 1e-8
    sharp_score /= total
    defocus_score /= total
    motion_score /= total

    return {
        "sharp_score": float(sharp_score),
        "defocus_score": float(defocus_score),
        "motion_score": float(motion_score),
        # 디버깅용 원본 특징값
        "features": {
            "vol": vol,
            "tenengrad": tenengrad,
            "hfr": hfr,
            "esw": esw,
            "rss": rss,
            "ai": ai,
            "str": str_val,
        }
    }


def _analyze_single_image(args: tuple) -> Tuple[str, Optional[dict]]:
    """
    단일 이미지를 분석하는 헬퍼 함수 (multiprocessing용).

    ProcessPoolExecutor와 함께 사용하기 위한 top-level 함수입니다.

    Args:
        args: (path, mode, tiles, params) 튜플

    Returns:
        (path, result_dict) 튜플, 실패 시 (path, None)
    """
    path, mode, tiles, params = args

    try:
        from .io_utils import imread_any

        img = imread_any(path)
        if img is None:
            return (path, None)

        # 그레이스케일 변환
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            return (path, None)

        if mode == "simple":
            # 간단 모드: 라플라시안 기반 빠른 분석
            try:
                lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                edge_mean = float(np.mean(np.abs(edge)))

                # 0-100 스케일로 정규화
                sharpness_score = min(100.0, lap / 5.0)
                edge_score = min(100.0, edge_mean / 0.1)

                # 가중 평균
                combined_score = 0.6 * sharpness_score + 0.4 * edge_score

                # 타입 판별 (간단한 임계값 기반)
                if combined_score > 60:
                    blur_type = "선명 ✅"
                    quality = "좋음"
                else:
                    # 방향성 체크 (간단 버전)
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
                    ang = (np.arctan2(gy, gx) + np.pi)
                    hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
                    direction_std = float(np.std(hist / (hist.sum() + 1e-8)))

                    if direction_std > 0.08:
                        blur_type = "모션블러 📸"
                        quality = "흐림 (움직임)"
                    else:
                        blur_type = "아웃포커스 🌫️"
                        quality = "흐림 (초점)"

                result = {
                    "score": round(combined_score, 1),
                    "type": blur_type,
                    "quality": quality,
                    "laplacian": round(lap, 2),
                    "edge": round(edge_mean, 2),
                    "direction": 0.0,
                }
                return (path, result)

            except Exception as e:
                print(f"Warning: Simple analysis failed for {path}: {e}")
                return (path, None)

        else:
            # 고급 모드: 다중 특징 기반 분석
            try:
                result = compute_scores_advanced(gray, tiles=tiles, params=params)
                return (path, result)
            except Exception as e:
                print(f"Warning: Advanced analysis failed for {path}: {e}")
                return (path, None)

    except Exception as e:
        print(f"Warning: Failed to process {path}: {e}")
        return (path, None)


def batch_analyze(
    paths: List[str],
    mode: str = "simple",
    tiles: int = 4,
    params: Optional[dict] = None,
    max_workers: int = 1
) -> Dict[str, dict]:
    """
    여러 이미지를 배치로 분석합니다.

    간단 모드(simple)는 라플라시안 기반의 빠른 분석을,
    고급 모드(advanced)는 더 정교한 다중 특징 분석을 수행합니다.

    Args:
        paths: 분석할 이미지 경로 리스트
        mode: "simple" 또는 "advanced"
        tiles: 타일 개수 (advanced 모드에서만 사용)
        params: 추가 분석 파라미터
        max_workers: 병렬 처리 워커 수 (1=순차, >1=병렬)

    Returns:
        {경로: 점수딕셔너리} 형태의 결과
    """
    if params is None:
        params = {}

    results = {}

    # 병렬 처리 또는 순차 처리 선택
    if max_workers > 1 and len(paths) > 1:
        # 병렬 처리 (CPU 집약적 작업이므로 ProcessPoolExecutor 사용)
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # 작업 인자 준비
            tasks = [(path, mode, tiles, params) for path in paths]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 작업 제출
                future_to_path = {
                    executor.submit(_analyze_single_image, task): task[0]
                    for task in tasks
                }

                # 결과 수집
                for future in as_completed(future_to_path):
                    try:
                        path, result = future.result()
                        if result is not None:
                            results[path] = result
                    except Exception as e:
                        path = future_to_path[future]
                        print(f"Warning: Failed to get result for {path}: {e}")

        except (ImportError, OSError) as e:
            # ProcessPoolExecutor 사용 불가 시 순차 처리로 폴백
            print(f"Warning: Parallel processing failed, falling back to sequential: {e}")
            max_workers = 1

    # 순차 처리 (max_workers=1 또는 폴백)
    if max_workers == 1:
        from .io_utils import imread_any

        for path in paths:
            try:
                img = imread_any(path)
                if img is None:
                    continue

                # 그레이스케일 변환
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                except Exception:
                    # 이미 그레이스케일이거나 변환 불가능
                    continue

                if mode == "simple":
                    # 간단 모드: 라플라시안 기반 빠른 분석
                    try:
                        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                        edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                        edge_mean = float(np.mean(np.abs(edge)))

                        # 0-100 스케일로 정규화
                        sharpness_score = min(100.0, lap / 5.0)
                        edge_score = min(100.0, edge_mean / 0.1)

                        # 가중 평균
                        combined_score = 0.6 * sharpness_score + 0.4 * edge_score

                        # 타입 판별 (간단한 임계값 기반)
                        if combined_score > 60:
                            blur_type = "선명 ✅"
                            quality = "좋음"
                        else:
                            # 방향성 체크 (간단 버전)
                            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                            mag = np.sqrt(gx*gx + gy*gy) + 1e-8
                            ang = (np.arctan2(gy, gx) + np.pi)
                            hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
                            direction_std = float(np.std(hist / (hist.sum() + 1e-8)))

                            if direction_std > 0.08:
                                blur_type = "모션블러 📸"
                                quality = "흐림 (움직임)"
                            else:
                                blur_type = "아웃포커스 🌫️"
                                quality = "흐림 (초점)"

                        results[path] = {
                            "score": round(combined_score, 1),
                            "type": blur_type,
                            "quality": quality,
                            "laplacian": round(lap, 2),
                            "edge": round(edge_mean, 2),
                            "direction": 0.0,
                        }

                    except Exception as e:
                        print(f"Warning: Simple analysis failed for {path}: {e}")
                        continue

                else:
                    # 고급 모드: 다중 특징 기반 분석
                    try:
                        results[path] = compute_scores_advanced(
                            gray,
                            tiles=tiles,
                            params=params
                        )
                    except Exception as e:
                        print(f"Warning: Advanced analysis failed for {path}: {e}")
                        continue

            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
                continue

    return results