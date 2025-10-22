from __future__ import annotations
import numpy as np
import cv2
from typing import Dict
from .features import (
    variance_of_laplacian, tenengrad, highfreq_ratio, edge_spread_width,
    anisotropy_index, radial_spectrum_slope, structure_tensor_ratio,
)
from .types import AdvancedScores

def _tile_features(gray: np.ndarray, tiles: int = 4) -> Dict[str, float]:
    H, W = gray.shape
    hs, ws = H//tiles, W//tiles
    vols, tens, hfrs, esws, anisos, slopes, strats = [], [], [], [], [], [], []
    for i in range(tiles):
        for j in range(tiles):
            crop = gray[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            if crop.size < 20:
                continue
            vols.append(variance_of_laplacian(crop))
            tens.append(tenengrad(crop))
            hfrs.append(highfreq_ratio(crop))
            esws.append(edge_spread_width(crop, sample_edges=60))
            anisos.append(anisotropy_index(crop))
            slopes.append(radial_spectrum_slope(crop))
            strats.append(structure_tensor_ratio(crop))
    def p(a, q): return float(np.percentile(a, q)) if a else 0.0
    return {
        "vol_p20":   p(vols,  20),
        "ten_p20":   p(tens,  20),
        "hfr_p20":   p(hfrs,  20),
        "esw_p80":   p(esws,  80),
        "aniso_p80": p(anisos, 80),
        "slope_p20": p(slopes, 20),
        "strat_p80": p(strats, 80),
    }

def _norm_box(x: float, lo: float, hi: float, invert: bool = False) -> float:
    v = (float(x) - lo) / (hi - lo + 1e-8)
    v = min(max(v, 0.0), 1.0)
    return 1.0 - v if invert else v

def compute_scores_advanced(gray: np.ndarray, *, tiles: int, params: Dict[str, float]) -> AdvancedScores:
    H, W = gray.shape
    long_side = int(params.get("long_side", 1024))
    if max(H, W) > long_side:
        s = long_side / max(H, W)
        gray = cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    F = _tile_features(gray, tiles=tiles)
    vol_n   = _norm_box(F["vol_p20"],   50,   600,  invert=False)
    ten_n   = _norm_box(F["ten_p20"],   1.0,  12.0, invert=False)
    hfr_n   = _norm_box(F["hfr_p20"],   0.02, 0.35, invert=False)
    esw_n   = _norm_box(F["esw_p80"],   1.0,  6.0,  invert=True)
    slope_n = _norm_box(F["slope_p20"], -6.0, -0.5, invert=True)
    aniso_n = _norm_box(F["aniso_p80"], 0.0,  0.12, invert=False)
    strat_n = _norm_box(F["strat_p80"], 0.02, 0.45, invert=False)

    sharp_score = (
        params.get("w_sharp_vol", 0.30) * vol_n +
        params.get("w_sharp_ten", 0.25) * ten_n +
        params.get("w_sharp_hfr", 0.20) * hfr_n +
        params.get("w_sharp_esw", 0.15) * esw_n +
        params.get("w_sharp_slope", 0.10) * slope_n
    )
    defocus_score = (
        params.get("w_def_esw", 0.40) * (1 - esw_n) +
        params.get("w_def_vol", 0.25) * (1 - vol_n) +
        params.get("w_def_slope", 0.25) * (1 - slope_n) +
        params.get("w_def_aniso", 0.10) * (1 - aniso_n)
    )
    motion_score = (
        params.get("w_mot_aniso", 0.60) * aniso_n +
        params.get("w_mot_strat", 0.30) * strat_n +
        params.get("w_mot_volinv", 0.10) * (1 - vol_n)
    )

    return AdvancedScores(
        features=F,
        normalized={
            "vol_n": vol_n, "ten_n": ten_n, "hfr_n": hfr_n,
            "esw_n": esw_n, "slope_n": slope_n, "aniso_n": aniso_n, "strat_n": strat_n
        },
        sharp_score=float(sharp_score),
        defocus_score=float(defocus_score),
        motion_score=float(motion_score),
    )
