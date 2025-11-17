"""
Unit tests for core analysis functions.

핵심 분석 함수들의 단위 테스트:
- list_images()
- load_thumbnail()
- batch_analyze()
- compute_scores_advanced()
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List
import unified_sort as us


class TestListImages:
    """list_images() 함수 테스트."""

    def test_list_images_nonexistent_dir(self):
        """존재하지 않는 디렉토리 처리 테스트."""
        result = us.list_images("/nonexistent/directory")
        assert result == []

    def test_list_images_finds_jpg(self, sample_sharp_image: Path):
        """JPG 이미지 검색 테스트."""
        parent_dir = sample_sharp_image.parent
        result = us.list_images(str(parent_dir))
        assert len(result) > 0
        assert any(str(sample_sharp_image) in r for r in result)

    def test_list_images_recursive(self, temp_test_dir: Path, sample_sharp_image: Path):
        """재귀적 검색 테스트."""
        # 서브디렉토리 생성
        sub_dir = temp_test_dir / "subdir"
        sub_dir.mkdir()

        # 서브디렉토리에 이미지 복사
        sub_image = sub_dir / "sub_image.jpg"
        import shutil
        shutil.copy(sample_sharp_image, sub_image)

        # 재귀 검색
        result = us.list_images(str(temp_test_dir), recursive=True)
        assert len(result) >= 2  # 원본 + 복사본

        # 비재귀 검색
        result_no_recursive = us.list_images(str(temp_test_dir), recursive=False)
        assert len(result) > len(result_no_recursive)


class TestLoadThumbnail:
    """load_thumbnail() 함수 테스트."""

    def test_load_thumbnail_success(self, sample_sharp_image: Path):
        """정상적인 썸네일 로딩 테스트."""
        thumb = us.load_thumbnail(str(sample_sharp_image), max_side=128)
        assert thumb is not None
        assert isinstance(thumb, np.ndarray)
        assert max(thumb.shape[:2]) <= 128

    def test_load_thumbnail_nonexistent_file(self):
        """존재하지 않는 파일 처리 테스트."""
        thumb = us.load_thumbnail("/nonexistent/image.jpg")
        assert thumb is None

    def test_load_thumbnail_resize(self, sample_sharp_image: Path):
        """리사이즈 크기 검증 테스트."""
        thumb_64 = us.load_thumbnail(str(sample_sharp_image), max_side=64)
        thumb_256 = us.load_thumbnail(str(sample_sharp_image), max_side=256)

        assert thumb_64 is not None
        assert thumb_256 is not None
        assert max(thumb_64.shape[:2]) <= 64
        assert max(thumb_256.shape[:2]) <= 256

    def test_load_thumbnail_aspect_ratio(self, temp_test_dir: Path):
        """종횡비 유지 테스트."""
        import cv2

        # 2:1 비율 이미지 생성
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        filepath = temp_test_dir / "aspect_test.jpg"
        cv2.imwrite(str(filepath), img)

        # 썸네일 로드
        thumb = us.load_thumbnail(str(filepath), max_side=100)
        assert thumb is not None

        h, w = thumb.shape[:2]
        # 종횡비가 대략 2:1 유지되어야 함
        assert abs(w / h - 2.0) < 0.1


class TestBatchAnalyze:
    """batch_analyze() 함수 테스트."""

    def test_batch_analyze_simple_mode(self, sample_images: Dict[str, Path]):
        """간단 모드 배치 분석 테스트."""
        paths = [str(p) for p in sample_images.values()]
        results = us.batch_analyze(paths, mode="simple")

        assert len(results) == len(paths)
        for path in paths:
            assert path in results
            assert "sharp_score" in results[path]
            assert "defocus_score" in results[path]
            assert "motion_score" in results[path]

    def test_batch_analyze_advanced_mode(self, sample_images: Dict[str, Path]):
        """고급 모드 배치 분석 테스트."""
        paths = [str(p) for p in sample_images.values()]
        results = us.batch_analyze(paths, mode="advanced", tiles=2)

        assert len(results) == len(paths)
        for path in paths:
            assert path in results
            scores = results[path]
            # 점수가 0~1 범위인지 확인
            assert 0 <= scores["sharp_score"] <= 1
            assert 0 <= scores["defocus_score"] <= 1
            assert 0 <= scores["motion_score"] <= 1

    @pytest.mark.slow
    def test_batch_analyze_multiprocessing(self, sample_image_batch: List[Path]):
        """멀티프로세싱 배치 분석 테스트."""
        paths = [str(p) for p in sample_image_batch]

        # 단일 프로세스
        results_single = us.batch_analyze(paths, mode="simple", max_workers=1)

        # 멀티 프로세스
        results_multi = us.batch_analyze(paths, mode="simple", max_workers=4)

        # 결과가 동일해야 함
        assert len(results_single) == len(results_multi)
        for path in paths:
            assert path in results_single
            assert path in results_multi

    def test_batch_analyze_empty_list(self):
        """빈 리스트 처리 테스트."""
        results = us.batch_analyze([])
        assert results == {}

    def test_batch_analyze_invalid_paths(self):
        """잘못된 경로 처리 테스트."""
        paths = ["/nonexistent1.jpg", "/nonexistent2.jpg"]
        results = us.batch_analyze(paths)
        # 잘못된 경로는 결과에서 제외되어야 함
        assert len(results) == 0


class TestComputeScoresAdvanced:
    """compute_scores_advanced() 함수 테스트."""

    def test_compute_scores_sharp_image(self, sample_sharp_image: Path):
        """선명한 이미지 점수 계산 테스트."""
        import cv2
        img = cv2.imread(str(sample_sharp_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scores = us.compute_scores_advanced(gray, tiles=2)

        # 선명한 이미지는 sharp_score가 높아야 함
        assert scores["sharp_score"] > scores["defocus_score"]
        assert scores["sharp_score"] > scores["motion_score"]

    def test_compute_scores_defocus_image(self, sample_defocus_image: Path):
        """디포커스 이미지 점수 계산 테스트."""
        import cv2
        img = cv2.imread(str(sample_defocus_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scores = us.compute_scores_advanced(gray, tiles=2)

        # 디포커스 이미지는 defocus_score가 높아야 함
        assert scores["defocus_score"] > scores["motion_score"]

    def test_compute_scores_output_format(self, sample_sharp_image: Path):
        """출력 형식 검증 테스트."""
        import cv2
        img = cv2.imread(str(sample_sharp_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scores = us.compute_scores_advanced(gray, tiles=4)

        # 필수 키 확인
        assert "sharp_score" in scores
        assert "defocus_score" in scores
        assert "motion_score" in scores

        # 점수 합이 1에 가까운지 확인
        total = scores["sharp_score"] + scores["defocus_score"] + scores["motion_score"]
        assert abs(total - 1.0) < 0.01

    def test_compute_scores_different_tile_counts(self, sample_sharp_image: Path):
        """다양한 타일 개수 테스트."""
        import cv2
        img = cv2.imread(str(sample_sharp_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for tiles in [1, 2, 4, 8]:
            scores = us.compute_scores_advanced(gray, tiles=tiles)
            assert "sharp_score" in scores
            assert 0 <= scores["sharp_score"] <= 1


class TestHelperFunctions:
    """헬퍼 함수 테스트."""

    def test_load_fullres(self, sample_sharp_image: Path):
        """전체 해상도 이미지 로딩 테스트."""
        img = us.load_fullres(str(sample_sharp_image))
        assert img is not None
        assert isinstance(img, np.ndarray)
        # 원본 크기 유지 (256x256)
        assert img.shape[0] == 256
        assert img.shape[1] == 256

    def test_phash_from_gray(self, sample_sharp_image: Path):
        """pHash 계산 테스트."""
        import cv2
        img = cv2.imread(str(sample_sharp_image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        phash = us.phash_from_gray(gray)
        assert isinstance(phash, int)
        assert phash >= 0

    def test_hamming_dist(self):
        """해밍 거리 계산 테스트."""
        hash1 = 0b1010101010101010
        hash2 = 0b1010101010101010
        hash3 = 0b0101010101010101

        # 동일한 해시
        assert us.hamming_dist(hash1, hash2) == 0

        # 모든 비트가 다른 해시
        assert us.hamming_dist(hash1, hash3) == 16

    def test_phash_similarity(self, sample_sharp_image: Path):
        """유사 이미지 pHash 테스트."""
        import cv2

        img1 = cv2.imread(str(sample_sharp_image))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # 약간 노이즈 추가
        img2 = img1 + np.random.randint(-10, 10, img1.shape, dtype=np.int16).astype(np.uint8)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        hash1 = us.phash_from_gray(gray1)
        hash2 = us.phash_from_gray(gray2)

        dist = us.hamming_dist(hash1, hash2)
        # 유사한 이미지는 해밍 거리가 작아야 함
        assert dist < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
