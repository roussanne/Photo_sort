from __future__ import annotations
from typing import Dict, Iterable, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from .io import load_image_bgr, to_gray
from .metrics import sharpness_simple
from .analysis import compute_scores_advanced

def batch_analyze(paths: Iterable[str], mode: Literal["simple","advanced"], *,
                  tiles: int = 4, params: Optional[dict] = None,
                  max_workers: Optional[int] = None) -> Dict[str, object]:
    params = params or {"long_side": 1024}
    if max_workers is None:
        max_workers = min(cpu_count(), 8)

    def _work(p: str):
        bgr = load_image_bgr(p, fullres=False, max_side=int(params.get("long_side", 1024)))
        if bgr is None:
            return None
        gray = to_gray(bgr)
        if mode == "simple":
            return p, sharpness_simple(gray)
        else:
            return p, compute_scores_advanced(gray, tiles=tiles, params=params)

    results: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_work, p): p for p in paths}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                k, v = res
                # dataclass라도 Streamlit에서 dict로 쓰고 싶으면 v.__dict__ 사용 가능
                results[k] = v if isinstance(v, dict) else getattr(v, "__dict__", v)
    return results
