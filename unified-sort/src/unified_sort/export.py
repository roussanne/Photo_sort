from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple
from .types import SimpleSharpness

def export_labeled_dataset(labels: Dict[str, str], out_root: str | Path,
                           *, move: bool = False) -> Tuple[int, Path]:
    out_root = Path(out_root)
    for lab in sorted(set(labels.values())):
        (out_root / lab).mkdir(parents=True, exist_ok=True)
    n_done = 0
    for src, lab in labels.items():
        dst = out_root / lab / Path(src).name
        try:
            if move:
                shutil.move(src, dst)
            else:
                shutil.copy2(src, dst)
            n_done += 1
        except Exception:
            pass
    return n_done, out_root

def move_or_delete_by_threshold(results: Dict[str, SimpleSharpness], *,
                                threshold: float, action: Literal["move","delete"],
                                dst_dir: Optional[str | Path] = None) -> int:
    count = 0
    if action == "move":
        assert dst_dir is not None, "dst_dir is required for move"
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
    for p, r in results.items():
        score = r["score"] if isinstance(r, dict) else r.score
        if score <= threshold:
            try:
                if action == "move":
                    shutil.move(p, Path(dst_dir) / Path(p).name)  # type: ignore
                else:
                    os.remove(p)
                count += 1
            except Exception:
                pass
    return count
