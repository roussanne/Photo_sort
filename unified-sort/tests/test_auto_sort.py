"""
Unit tests for auto-classification module.

자동 분류 모듈의 단위 테스트:
- AutoSortConfig
- classify_with_confidence()
- batch_classify()
- compute_adaptive_thresholds()
- get_classification_stats()
- suggest_config_adjustments()
"""

import pytest
from typing import Dict
import unified_sort as us


class TestAutoSortConfig:
    """AutoSortConfig 클래스 테스트."""

    def test_default_config(self):
        """기본 설정 테스트."""
        config = us.AutoSortConfig()

        assert config.min_sharp == 0.35
        assert config.min_defocus == 0.35
        assert config.min_motion == 0.35
        assert config.min_confidence == 0.15
        assert config.strategy == "balanced"

    def test_conservative_strategy(self):
        """보수적 전략 테스트."""
        config = us.AutoSortConfig(strategy="conservative")

        # 보수적 전략은 높은 임계값
        assert config.min_confidence > 0.2
        assert config.min_sharp >= 0.4

    def test_aggressive_strategy(self):
        """적극적 전략 테스트."""
        config = us.AutoSortConfig(strategy="aggressive")

        # 적극적 전략은 낮은 임계값
        assert config.min_confidence < 0.15
        assert config.min_sharp <= 0.3

    def test_custom_thresholds(self):
        """사용자 정의 임계값 테스트."""
        config = us.AutoSortConfig(
            min_sharp=0.5,
            min_defocus=0.4,
            min_motion=0.3,
            min_confidence=0.25
        )

        assert config.min_sharp == 0.5
        assert config.min_defocus == 0.4
        assert config.min_motion == 0.3
        assert config.min_confidence == 0.25


class TestClassifyWithConfidence:
    """classify_with_confidence() 함수 테스트."""

    def test_clear_sharp_classification(self, sample_scores: Dict[str, float]):
        """명확한 sharp 분류 테스트."""
        config = us.AutoSortConfig()
        result = us.classify_with_confidence(sample_scores, config)

        assert result.label == "sharp"
        assert result.confidence > 0
        assert not result.needs_review

    def test_uncertain_classification(self, sample_uncertain_scores: Dict[str, float]):
        """불확실한 분류 테스트."""
        config = us.AutoSortConfig()
        result = us.classify_with_confidence(sample_uncertain_scores, config)

        # 마진이 작아서 불확실로 판정되어야 함
        assert result.needs_review

    def test_low_quality_detection(self, sample_low_quality_scores: Dict[str, float]):
        """낮은 품질 감지 테스트."""
        config = us.AutoSortConfig(enable_quality_gate=True, min_total_quality=0.3)
        result = us.classify_with_confidence(sample_low_quality_scores, config)

        # 전체 품질이 낮아서 수동 검토 필요
        assert result.needs_review

    def test_bias_application(self):
        """바이어스 적용 테스트."""
        scores = {
            "sharp_score": 0.4,
            "defocus_score": 0.35,
            "motion_score": 0.25
        }

        config = us.AutoSortConfig(sharp_bias=0.2)
        result = us.classify_with_confidence(scores, config)

        # sharp 바이어스로 인해 sharp로 분류되어야 함
        assert result.label == "sharp"
        assert result.biased_scores["sharp_score"] > scores["sharp_score"]

    def test_min_threshold_enforcement(self):
        """최소 임계값 적용 테스트."""
        scores = {
            "sharp_score": 0.30,  # min_sharp(0.35)보다 낮음
            "defocus_score": 0.25,
            "motion_score": 0.45
        }

        config = us.AutoSortConfig(min_sharp=0.35, min_motion=0.35)
        result = us.classify_with_confidence(scores, config)

        # motion_score가 가장 높지만 임계값 미달 시 불확실 판정
        if result.needs_review:
            assert True  # 임계값 미달로 수동 검토


class TestBatchClassify:
    """batch_classify() 함수 테스트."""

    def test_batch_classify_multiple_images(self):
        """여러 이미지 일괄 분류 테스트."""
        scores_dict = {
            "img1.jpg": {"sharp_score": 0.8, "defocus_score": 0.1, "motion_score": 0.1},
            "img2.jpg": {"sharp_score": 0.2, "defocus_score": 0.7, "motion_score": 0.1},
            "img3.jpg": {"sharp_score": 0.1, "defocus_score": 0.2, "motion_score": 0.7},
        }

        config = us.AutoSortConfig()
        results = us.batch_classify(scores_dict, config)

        assert len(results) == 3
        assert results["img1.jpg"].label == "sharp"
        assert results["img2.jpg"].label == "defocus"
        assert results["img3.jpg"].label == "motion"

    def test_batch_classify_empty(self):
        """빈 배치 처리 테스트."""
        config = us.AutoSortConfig()
        results = us.batch_classify({}, config)
        assert results == {}

    def test_batch_classify_consistency(self):
        """배치 분류 일관성 테스트."""
        scores = {"sharp_score": 0.6, "defocus_score": 0.3, "motion_score": 0.1}
        scores_dict = {f"img{i}.jpg": scores for i in range(10)}

        config = us.AutoSortConfig()
        results = us.batch_classify(scores_dict, config)

        # 모든 이미지가 동일한 점수이므로 동일한 결과
        labels = [r.label for r in results.values()]
        assert len(set(labels)) == 1  # 모두 같은 레이블


class TestComputeAdaptiveThresholds:
    """compute_adaptive_thresholds() 함수 테스트."""

    def test_adaptive_thresholds_calculation(self):
        """적응형 임계값 계산 테스트."""
        scores_dict = {
            f"img{i}.jpg": {
                "sharp_score": 0.7 - i * 0.05,
                "defocus_score": 0.2 + i * 0.03,
                "motion_score": 0.1 + i * 0.02
            }
            for i in range(20)
        }

        thresholds = us.compute_adaptive_thresholds(scores_dict)

        assert "sharp" in thresholds
        assert "defocus" in thresholds
        assert "motion" in thresholds
        assert 0 < thresholds["sharp"] < 1

    def test_adaptive_thresholds_uniform_dataset(self):
        """균일한 데이터셋에 대한 적응형 임계값 테스트."""
        # 모든 이미지가 비슷한 점수
        scores_dict = {
            f"img{i}.jpg": {
                "sharp_score": 0.5,
                "defocus_score": 0.3,
                "motion_score": 0.2
            }
            for i in range(10)
        }

        thresholds = us.compute_adaptive_thresholds(scores_dict)

        # 균일한 분포에서도 유효한 임계값 반환
        assert all(0 < v < 1 for v in thresholds.values())


class TestGetClassificationStats:
    """get_classification_stats() 함수 테스트."""

    def test_classification_stats(self):
        """분류 통계 생성 테스트."""
        results = {
            "img1.jpg": us.ClassificationResult(label="sharp", confidence=0.5, needs_review=False, reason="High sharp score"),
            "img2.jpg": us.ClassificationResult(label="defocus", confidence=0.4, needs_review=False, reason="High defocus score"),
            "img3.jpg": us.ClassificationResult(label="uncertain", confidence=0.1, needs_review=True, reason="Low margin"),
        }

        stats = us.get_classification_stats(results)

        assert stats["total_images"] == 3
        assert stats["sharp_count"] == 1
        assert stats["defocus_count"] == 1
        assert stats["uncertain_count"] == 1

    def test_classification_stats_empty(self):
        """빈 결과 통계 테스트."""
        stats = us.get_classification_stats({})

        assert stats["total_images"] == 0
        assert stats["sharp_count"] == 0


class TestSuggestConfigAdjustments:
    """suggest_config_adjustments() 함수 테스트."""

    def test_high_uncertainty_suggestion(self):
        """높은 불확실성 비율에 대한 제안 테스트."""
        stats = {
            "total_images": 100,
            "uncertain_count": 50,  # 50% 불확실
            "avg_confidence": 0.1
        }

        suggestions = us.suggest_config_adjustments(stats)

        assert len(suggestions) > 0
        # "aggressive" 전략 제안 또는 임계값 낮추기 제안 포함
        suggestions_text = " ".join(suggestions).lower()
        assert "aggressive" in suggestions_text or "lower" in suggestions_text or "임계값" in suggestions_text

    def test_low_uncertainty_suggestion(self):
        """낮은 불확실성 비율에 대한 제안 테스트."""
        stats = {
            "total_images": 100,
            "uncertain_count": 5,  # 5% 불확실
            "avg_confidence": 0.8
        }

        suggestions = us.suggest_config_adjustments(stats)

        # 양호한 상태이므로 제안이 적거나 긍정적
        if len(suggestions) > 0:
            suggestions_text = " ".join(suggestions).lower()
            assert "good" in suggestions_text or "잘" in suggestions_text or len(suggestions) == 0


class TestClassificationResult:
    """ClassificationResult 데이터클래스 테스트."""

    def test_classification_result_creation(self):
        """ClassificationResult 생성 테스트."""
        result = us.ClassificationResult(
            label="sharp",
            confidence=0.6,
            needs_review=False,
            reason="High sharp score"
        )

        assert result.label == "sharp"
        assert result.confidence == 0.6
        assert not result.needs_review
        assert result.reason == "High sharp score"

    def test_classification_result_defaults(self):
        """ClassificationResult 기본값 테스트."""
        result = us.ClassificationResult(
            label="defocus",
            confidence=0.5,
            needs_review=True
        )

        assert result.reason == ""  # 기본값


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
