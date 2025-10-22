"""
Helpers: high-res loader, pHash, hamming, widget key.
"""
import hashlib
from typing import Callable
import numpy as np
import cv2

def load_fullres(path: str, max_side: int | None = 2048):
    from .io_utils import imread_any
    try:
        img_bgr = imread_any(path)
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception:
        return None

def phash_from_gray(gray: np.ndarray, hash_size: int = 8) -> int:
    g = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = g[:, 1:] > g[:, :-1]
    return sum(1 << i for (i, v) in enumerate(diff.flatten()) if v)

def hamming_dist(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def make_widget_key(prefix: str, path: str) -> str:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"
