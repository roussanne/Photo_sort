"""
Advanced auto-sorting and classification module with confidence scoring.

This module provides precise, configurable auto-labeling with:
- Confidence-based classification
- Margin analysis for uncertainty detection
- Adaptive thresholds
- User fine-tuning controls
- Multi-strategy decision making

개선사항:
1. 신뢰도 점수 계산 (상위 2개 클래스 간 마진)
2. 불확실성 감지 (수동 검토 필요 항목 표시)
3. 적응형 임계값 (데이터셋 통계 기반)
4. 사용자 조정 가능한 파라미터
5. 정밀도/재현율 트레이드오프 제어
"""

from typing import Dict, List, Tuple, Optional, Literal
import numpy as np
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """
    분류 결과와 신뢰도 정보를 담는 데이터 클래스.

    Attributes:
        label: 예측된 라벨 ("sharp", "defocus", "motion", "uncertain")
        confidence: 신뢰도 점수 (0~1, 높을수록 확실함)
        scores: 각 클래스별 원본 점수
        margin: 1등과 2등 점수 차이 (높을수록 명확함)
        needs_review: 수동 검토 필요 여부
        alternative_label: 2순위 라벨
        reasoning: 분류 근거 설명
    """
    label: str
    confidence: float
    scores: Dict[str, float]
    margin: float
    needs_review: bool
    alternative_label: Optional[str] = None
    reasoning: str = ""


class AutoSortConfig:
    """
    자동 분류 설정을 관리하는 클래스.

    사용자가 조정 가능한 모든 파라미터를 중앙 집중식으로 관리합니다.
    """

    def __init__(
        self,
        # 절대 임계값 (각 클래스의 최소 점수)
        min_sharp: float = 0.35,
        min_defocus: float = 0.35,
        min_motion: float = 0.35,

        # 신뢰도 설정
        min_confidence: float = 0.15,  # 최소 마진 (1등-2등)
        uncertainty_threshold: float = 0.10,  # 이보다 작으면 불확실

        # 분류 전략
        strategy: Literal["conservative", "balanced", "aggressive"] = "balanced",

        # 클래스 바이어스 (기본값에서 조정, -0.2 ~ +0.2)
        sharp_bias: float = 0.0,
        defocus_bias: float = 0.0,
        motion_bias: float = 0.0,

        # 품질 게이팅 (전체적으로 낮은 점수 거부)
        min_total_quality: float = 0.20,

        # 적응형 설정
        use_adaptive_thresholds: bool = False,
        adaptive_percentile: float = 0.3,  # 하위 30%를 기준으로
    ):
        """
        자동 분류 설정 초기화.

        Args:
            min_sharp: 선명 최소 점수
            min_defocus: 아웃포커스 최소 점수
            min_motion: 모션블러 최소 점수
            min_confidence: 1등과 2등 간 최소 마진
            uncertainty_threshold: 불확실 판정 임계값
            strategy: 분류 전략 (보수적/균형/적극적)
            sharp_bias: 선명 점수 바이어스
            defocus_bias: 아웃포커스 점수 바이어스
            motion_bias: 모션블러 점수 바이어스
            min_total_quality: 최소 전체 품질 점수
            use_adaptive_thresholds: 적응형 임계값 사용 여부
            adaptive_percentile: 적응형 임계값 백분위수
        """
        self.min_sharp = min_sharp
        self.min_defocus = min_defocus
        self.min_motion = min_motion
        self.min_confidence = min_confidence
        self.uncertainty_threshold = uncertainty_threshold
        self.strategy = strategy
        self.sharp_bias = sharp_bias
        self.defocus_bias = defocus_bias
        self.motion_bias = motion_bias
        self.min_total_quality = min_total_quality
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.adaptive_percentile = adaptive_percentile

        # 전략별 설정 조정
        self._apply_strategy()

    def _apply_strategy(self):
        """전략에 따라 임계값을 자동 조정합니다."""
        if self.strategy == "conservative":
            # 보수적: 높은 신뢰도 요구, 불확실하면 수동 검토
            self.min_confidence = max(self.min_confidence, 0.20)
            self.uncertainty_threshold = max(self.uncertainty_threshold, 0.15)
        elif self.strategy == "aggressive":
            # 적극적: 낮은 신뢰도에서도 분류
            self.min_confidence = min(self.min_confidence, 0.08)
            self.uncertainty_threshold = min(self.uncertainty_threshold, 0.05)
        # balanced는 사용자 설정 그대로 사용

    def get_class_threshold(self, class_name: str) -> float:
        """클래스별 임계값 반환 (바이어스 적용)."""
        thresholds = {
            "sharp": self.min_sharp + self.sharp_bias,
            "defocus": self.min_defocus + self.defocus_bias,
            "motion": self.min_motion + self.motion_bias,
        }
        return max(0.0, min(1.0, thresholds.get(class_name, 0.35)))


def apply_bias_to_scores(
    scores: Dict[str, float],
    config: AutoSortConfig
) -> Dict[str, float]:
    """
    점수에 바이어스를 적용합니다.

    사용자가 특정 클래스를 선호하거나 억제하도록 설정할 수 있습니다.

    Args:
        scores: 원본 점수 딕셔너리
        config: 설정 객체

    Returns:
        바이어스가 적용된 점수 딕셔너리
    """
    biased = dict(scores)
    biased["sharp_score"] = scores.get("sharp_score", 0.0) + config.sharp_bias
    biased["defocus_score"] = scores.get("defocus_score", 0.0) + config.defocus_bias
    biased["motion_score"] = scores.get("motion_score", 0.0) + config.motion_bias

    # 0~1 범위로 클램핑
    for key in biased:
        biased[key] = max(0.0, min(1.0, biased[key]))

    return biased


def calculate_confidence_and_margin(
    scores: Dict[str, float]
) -> Tuple[str, str, float, float]:
    """
    점수를 분석하여 1등, 2등 클래스와 신뢰도를 계산합니다.

    Args:
        scores: 클래스별 점수 딕셔너리

    Returns:
        (1등 클래스, 2등 클래스, 마진, 1등 점수) 튜플
    """
    # 점수 추출
    sharp = scores.get("sharp_score", 0.0)
    defocus = scores.get("defocus_score", 0.0)
    motion = scores.get("motion_score", 0.0)

    # 점수 정렬
    class_scores = [
        ("sharp", sharp),
        ("defocus", defocus),
        ("motion", motion),
    ]
    class_scores.sort(key=lambda x: x[1], reverse=True)

    top_class, top_score = class_scores[0]
    second_class, second_score = class_scores[1]

    # 마진 계산 (1등과 2등 차이)
    margin = top_score - second_score

    return top_class, second_class, margin, top_score


def classify_with_confidence(
    scores: Dict[str, float],
    config: AutoSortConfig
) -> ClassificationResult:
    """
    점수를 분석하여 신뢰도 기반 분류를 수행합니다.

    이 함수는 단순 argmax가 아닌 다층 결정 로직을 사용합니다:
    1. 바이어스 적용
    2. 최고 점수 클래스 선택
    3. 최소 임계값 검증
    4. 마진/신뢰도 검증
    5. 전체 품질 게이팅
    6. 불확실성 감지

    Args:
        scores: 원본 점수 딕셔너리
        config: 설정 객체

    Returns:
        ClassificationResult 객체
    """
    # 1단계: 바이어스 적용
    biased_scores = apply_bias_to_scores(scores, config)

    # 2단계: 신뢰도 분석
    top_class, second_class, margin, top_score = calculate_confidence_and_margin(
        biased_scores
    )

    # 3단계: 전체 품질 검증
    total_quality = sum([
        biased_scores.get("sharp_score", 0.0),
        biased_scores.get("defocus_score", 0.0),
        biased_scores.get("motion_score", 0.0)
    ])

    if total_quality < config.min_total_quality:
        return ClassificationResult(
            label="uncertain",
            confidence=0.0,
            scores=scores,
            margin=margin,
            needs_review=True,
            alternative_label=None,
            reasoning="전체 품질 점수가 너무 낮음 (분석 실패 가능성)"
        )

    # 4단계: 클래스별 최소 임계값 검증
    min_threshold = config.get_class_threshold(top_class)

    if top_score < min_threshold:
        # 1등이 임계값 미달 -> 2등으로 재할당 시도
        second_threshold = config.get_class_threshold(second_class)
        second_score = biased_scores[f"{second_class}_score"]

        if second_score >= second_threshold:
            # 2등이 임계값 통과
            return ClassificationResult(
                label=second_class,
                confidence=margin * 0.7,  # 재할당이므로 신뢰도 감소
                scores=scores,
                margin=margin,
                needs_review=margin < config.uncertainty_threshold,
                alternative_label=top_class,
                reasoning=f"{top_class}가 1등이지만 임계값 미달, {second_class}로 재할당"
            )
        else:
            # 둘 다 임계값 미달
            return ClassificationResult(
                label="uncertain",
                confidence=0.0,
                scores=scores,
                margin=margin,
                needs_review=True,
                alternative_label=top_class,
                reasoning="상위 클래스들이 모두 최소 임계값 미달"
            )

    # 5단계: 신뢰도 검증
    if margin < config.uncertainty_threshold:
        # 마진이 너무 작음 -> 불확실
        return ClassificationResult(
            label="uncertain",
            confidence=margin,
            scores=scores,
            margin=margin,
            needs_review=True,
            alternative_label=second_class,
            reasoning=f"{top_class}와 {second_class} 점수 차이가 너무 작음 (마진: {margin:.3f})"
        )

    # 6단계: 정상 분류 성공
    needs_review = margin < config.min_confidence

    return ClassificationResult(
        label=top_class,
        confidence=margin,
        scores=scores,
        margin=margin,
        needs_review=needs_review,
        alternative_label=second_class if needs_review else None,
        reasoning=f"명확한 분류 (마진: {margin:.3f}, 1등 점수: {top_score:.3f})"
    )


def batch_classify(
    scores_dict: Dict[str, Dict[str, float]],
    config: AutoSortConfig
) -> Dict[str, ClassificationResult]:
    """
    여러 이미지를 배치로 분류합니다.

    Args:
        scores_dict: {경로: 점수딕셔너리} 형태
        config: 설정 객체

    Returns:
        {경로: ClassificationResult} 딕셔너리
    """
    results = {}
    for path, scores in scores_dict.items():
        results[path] = classify_with_confidence(scores, config)
    return results


def compute_adaptive_thresholds(
    scores_dict: Dict[str, Dict[str, float]],
    percentile: float = 0.3
) -> Dict[str, float]:
    """
    데이터셋 통계를 기반으로 적응형 임계값을 계산합니다.

    하위 N% 점수를 기준으로 임계값을 설정하여,
    데이터셋 특성에 맞게 자동 조정합니다.

    Args:
        scores_dict: 전체 이미지의 점수 딕셔너리
        percentile: 백분위수 (0~1, 기본 0.3 = 하위 30%)

    Returns:
        {"sharp": threshold, "defocus": threshold, "motion": threshold}
    """
    sharp_scores = []
    defocus_scores = []
    motion_scores = []

    for scores in scores_dict.values():
        sharp_scores.append(scores.get("sharp_score", 0.0))
        defocus_scores.append(scores.get("defocus_score", 0.0))
        motion_scores.append(scores.get("motion_score", 0.0))

    if not sharp_scores:
        return {"sharp": 0.35, "defocus": 0.35, "motion": 0.35}

    # 각 클래스별 백분위수 계산
    thresholds = {
        "sharp": float(np.percentile(sharp_scores, percentile * 100)),
        "defocus": float(np.percentile(defocus_scores, percentile * 100)),
        "motion": float(np.percentile(motion_scores, percentile * 100)),
    }

    # 최소값 보장 (너무 낮은 임계값 방지)
    for key in thresholds:
        thresholds[key] = max(0.15, min(0.6, thresholds[key]))

    return thresholds


def get_classification_stats(
    results: Dict[str, ClassificationResult]
) -> Dict[str, any]:
    """
    분류 결과의 통계를 계산합니다.

    Args:
        results: batch_classify의 결과

    Returns:
        통계 딕셔너리 (클래스별 개수, 불확실 개수, 평균 신뢰도 등)
    """
    total = len(results)
    if total == 0:
        return {}

    sharp_count = sum(1 for r in results.values() if r.label == "sharp")
    defocus_count = sum(1 for r in results.values() if r.label == "defocus")
    motion_count = sum(1 for r in results.values() if r.label == "motion")
    uncertain_count = sum(1 for r in results.values() if r.label == "uncertain")
    needs_review_count = sum(1 for r in results.values() if r.needs_review)

    confidences = [r.confidence for r in results.values() if r.confidence > 0]
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    median_confidence = float(np.median(confidences)) if confidences else 0.0

    margins = [r.margin for r in results.values()]
    avg_margin = float(np.mean(margins)) if margins else 0.0

    return {
        "total": total,
        "sharp": sharp_count,
        "defocus": defocus_count,
        "motion": motion_count,
        "uncertain": uncertain_count,
        "needs_review": needs_review_count,
        "avg_confidence": avg_confidence,
        "median_confidence": median_confidence,
        "avg_margin": avg_margin,
        "sharp_pct": (sharp_count / total * 100) if total > 0 else 0,
        "defocus_pct": (defocus_count / total * 100) if total > 0 else 0,
        "motion_pct": (motion_count / total * 100) if total > 0 else 0,
        "uncertain_pct": (uncertain_count / total * 100) if total > 0 else 0,
        "review_pct": (needs_review_count / total * 100) if total > 0 else 0,
    }


def suggest_config_adjustments(
    stats: Dict[str, any]
) -> List[str]:
    """
    통계를 분석하여 설정 조정 제안을 생성합니다.

    Args:
        stats: get_classification_stats의 결과

    Returns:
        제안 메시지 리스트
    """
    suggestions = []

    # 불확실 항목이 너무 많음
    if stats.get("uncertain_pct", 0) > 20:
        suggestions.append(
            "⚠️ 불확실 항목이 20% 이상입니다. "
            "전략을 'aggressive'로 변경하거나 최소 임계값을 낮춰보세요."
        )

    # 검토 필요 항목이 너무 많음
    if stats.get("review_pct", 0) > 30:
        suggestions.append(
            "💡 수동 검토 필요 항목이 30% 이상입니다. "
            "min_confidence를 낮추거나 적응형 임계값을 활성화하세요."
        )

    # 평균 신뢰도가 낮음
    if stats.get("avg_confidence", 0) < 0.15:
        suggestions.append(
            "📊 평균 신뢰도가 낮습니다. "
            "이미지 품질이 전반적으로 애매하거나, 분석 파라미터 조정이 필요할 수 있습니다."
        )

    # 한 클래스가 지배적
    for class_name in ["sharp", "defocus", "motion"]:
        pct = stats.get(f"{class_name}_pct", 0)
        if pct > 70:
            suggestions.append(
                f"🎯 {class_name} 클래스가 70% 이상입니다. "
                f"다른 클래스의 바이어스를 높이거나 {class_name}_bias를 낮춰보세요."
            )

    # 모든 것이 정상
    if not suggestions:
        suggestions.append("✅ 분류 통계가 양호합니다!")

    return suggestions
