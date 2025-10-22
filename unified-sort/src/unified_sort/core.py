"""
Core analysis stubs.
Replace with your existing implementation if you already have it.
"""
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2

def list_images(root: str, recursive: bool = False) -> List[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    pats = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
            "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
    paths = []
    if recursive:
        for p in pats:
            paths += list(root_path.rglob(p))
    else:
        for p in pats:
            paths += list(root_path.glob(p))
    return [str(p) for p in sorted(set([p for p in paths if p.is_file()]))]

def load_thumbnail(path: str, max_side: int = 384):
    """Return BGR ndarray thumbnail, or None on failure."""
    from .io_utils import imread_any
    img = imread_any(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def compute_scores_advanced(gray: np.ndarray, tiles: int, params: dict) -> dict:
    """Dummy advanced scoring. Replace with your full implementation."""
    # Minimal placeholder: return fixed structure so UI works.
    return {
        "sharp_score": 0.5,
        "defocus_score": 0.3,
        "motion_score": 0.2,
    }

def batch_analyze(paths: List[str], mode: str = "simple", tiles: int = 4, params: dict = None, max_workers: int = 1) -> Dict[str, dict]:
    from .io_utils import imread_any
    res = {}
    for p in paths:
        img = imread_any(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mode == "simple":
            # Very simple Laplacian based score (0-100)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            edge_mean = float(np.mean(np.abs(edge)))
            sharpness_score = min(100.0, lap / 5.0)
            edge_score = min(100.0, edge_mean / 0.1)
            score = 0.6 * sharpness_score + 0.4 * edge_score
            blur_type = "선명 ✅" if score > 60 else "아웃포커스 🌫️"
            res[p] = {
                "score": round(score, 1),
                "type": blur_type,
                "quality": "좋음" if score > 60 else "흐림 (초점)",
                "laplacian": round(lap, 2),
                "edge": round(edge_mean, 2),
                "direction": 0.0,
            }
        else:
            res[p] = compute_scores_advanced(gray, tiles=tiles, params=params or {})
    return res
