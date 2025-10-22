from __future__ import annotations
import io
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from PIL import Image

# 선택 기능 플래그
try:
    import pillow_heif  # type: ignore
    USE_HEIC = True
except Exception:
    USE_HEIC = False

try:
    import rawpy  # type: ignore
    USE_RAWPY = True
except Exception:
    USE_RAWPY = False

def list_images(root: str | Path, recursive: bool = False) -> List[str]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    patterns = [
        "*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
        "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP",
    ]
    if USE_HEIC:
        patterns += ["*.heic","*.heif","*.HEIC","*.HEIF"]
    if USE_RAWPY:
        patterns += ["*.rw2","*.RW2"]

    paths: list[Path] = []
    if recursive:
        for pat in patterns:
            paths += list(root_path.rglob(pat))
    else:
        for pat in patterns:
            paths += list(root_path.glob(pat))
    paths = [p for p in paths if p.is_file()]
    return [str(p) for p in sorted(set(paths))]

def _load_bgr_general(p: str) -> Optional[np.ndarray]:
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def load_image_bgr(path: str, *, fullres: bool = False, max_side: Optional[int] = None) -> Optional[np.ndarray]:
    p = str(path)
    ext = p.lower().split(".")[-1]

    # HEIC/HEIF
    if USE_HEIC and ext in ("heic", "heif"):
        try:
            heif = pillow_heif.read_heif(p)
            img = Image.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = None

    # RW2
    elif USE_RAWPY and ext == "rw2":
        try:
            with rawpy.imread(p) as raw:
                if not fullres:
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
                            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        else:
                            rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True,
                                                  gamma=(2.2, 4.5), half_size=True, output_bps=8)
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    except Exception:
                        rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True,
                                              gamma=(2.2, 4.5), half_size=True, output_bps=8)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True,
                                          gamma=(2.2, 4.5), half_size=False, output_bps=8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = None
    else:
        bgr = _load_bgr_general(p)

    if bgr is not None and max_side:
        h, w = bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return bgr

def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
