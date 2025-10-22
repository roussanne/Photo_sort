from __future__ import annotations
import numpy as np
import cv2
from .types import SimpleSharpness

def sharpness_simple(gray: np.ndarray) -> SimpleSharpness:
    h, w = gray.shape
    if max(h, w) > 1024:
        s = 1024 / max(h, w)
        gray = cv2.resize(gray, (int(w*s), int(h*s)))

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = float(np.mean(np.sqrt(gx*gx + gy*gy)))

    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)
    hist, _ = np.histogram(ang, bins=18, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    direction = float(np.std(hist))

    s1 = min(100.0, (lap_var / 5.0))
    s2 = min(100.0, (edge / 0.1))
    score = (s1 * 0.6 + s2 * 0.4)

    if score > 60:
        blur_type = "선명 ✅"; quality = "좋음"
    elif direction > 0.08:
        blur_type = "모션블러 📸"; quality = "흐림 (움직임)"
    else:
        blur_type = "아웃포커스 🌫️"; quality = "흐림 (초점)"

    return SimpleSharpness(
        score=round(float(score), 1),
        type=blur_type, quality=quality,
        laplacian=round(float(lap_var), 2), edge=round(edge, 2), direction=round(direction, 3)
    )
