from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class SimpleSharpness:
    score: float
    type: str          # "선명 ✅" | "아웃포커스 🌫️" | "모션블러 📸"
    quality: str       # "좋음" | "흐림 (초점)" | "흐림 (움직임)"
    laplacian: float
    edge: float
    direction: float

@dataclass
class AdvancedScores:
    features: Dict[str, float]
    normalized: Dict[str, float]
    sharp_score: float
    defocus_score: float
    motion_score: float
