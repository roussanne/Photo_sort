from __future__ import annotations
from typing import Optional
import numpy as np
from .io import load_image_bgr

def load_thumbnail(path: str, max_side: int = 384) -> Optional[np.ndarray]:
    return load_image_bgr(path, fullres=False, max_side=max_side)

def load_fullres(path: str, max_side: int = 2048) -> Optional[np.ndarray]:
    return load_image_bgr(path, fullres=True, max_side=max_side)
