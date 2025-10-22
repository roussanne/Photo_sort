"""
IO utilities: imread_any and dataset export.
"""
from pathlib import Path
import shutil
import numpy as np
import cv2

try:
    import pillow_heif  # type: ignore
    _USE_HEIC = True
except Exception:
    pillow_heif = None
    _USE_HEIC = False

def imread_any(path: str):
    p = str(path)
    ext = p.lower().split(".")[-1]
    if _USE_HEIC and ext in ("heic", "heif"):
        try:
            heif = pillow_heif.read_heif(p)
            from PIL import Image as _PILImage
            img = _PILImage.frombytes(heif.mode, heif.size, heif.data, "raw").convert("RGB")
            import numpy as _np
            return cv2.cvtColor(_np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    try:
        data = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        return data
    except Exception:
        return None

def export_labeled_dataset(labels: dict, out_root: Path, move: bool = False):
    out_root = Path(out_root)
    (out_root / "sharp").mkdir(parents=True, exist_ok=True)
    (out_root / "defocus").mkdir(parents=True, exist_ok=True)
    (out_root / "motion").mkdir(parents=True, exist_ok=True)
    n = 0
    for p, lab in labels.items():
        dst = out_root / lab / Path(p).name
        try:
            if move:
                shutil.move(p, dst)
            else:
                shutil.copy2(p, dst)
            n += 1
        except Exception:
            pass
    return n, out_root
