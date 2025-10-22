from __future__ import annotations
import numpy as np
import cv2

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx*gx + gy*gy)))

def highfreq_ratio(gray: np.ndarray, cutoff: float = 0.1) -> float:
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = int(min(h, w) * cutoff)
    mask = (X-cx)**2 + (Y-cy)**2 > r*r
    total = np.sum(np.abs(fshift))
    high = np.sum(np.abs(fshift[mask]))
    return float(high / (total + 1e-8))

def radial_spectrum_slope(gray: np.ndarray, cutoff: float = 0.6) -> float:
    img = gray.astype(np.float32)
    wy = np.hanning(img.shape[0])[:, None]
    wx = np.hanning(img.shape[1])[None, :]
    win = wy * wx
    f = np.fft.fftshift(np.fft.fft2(img * win))
    mag = np.abs(f)

    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    r = R.astype(np.int32)
    rmax = int(min(h, w) * 0.5 * cutoff)

    bins = []
    for rad in range(1, rmax):
        mask = (r == rad)
        if np.any(mask):
            bins.append(np.mean(mag[mask]))
    bins = np.array(bins, dtype=np.float32)
    if len(bins) < 8:
        return 0.0

    x = np.arange(len(bins), dtype=np.float32)
    y = np.log(bins + 1e-8)
    start = len(x)//2
    Xmat = np.vstack([x[start:], np.ones_like(x[start:])]).T
    slope, _ = np.linalg.lstsq(Xmat, y[start:], rcond=None)[0]
    return float(slope)

def anisotropy_index(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)
    nbins = 18
    hist, _ = np.histogram(ang, bins=nbins, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return float(np.std(hist))

def structure_tensor_ratio(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), 1.0)
    Jyy = cv2.GaussianBlur(gy*gy, (0,0), 1.0)
    Jxy = cv2.GaussianBlur(gx*gy, (0,0), 1.0)
    l1 = 0.5*((Jxx+Jyy) + np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    l2 = 0.5*((Jxx+Jyy) - np.sqrt((Jxx-Jyy)**2 + 4*Jxy*Jxy))
    num = (l1 - l2)
    den = (l1 + l2 + 1e-8)
    return float(np.mean(num/den))

def edge_spread_width(gray: np.ndarray, sample_edges: int = 150) -> float:
    edges = cv2.Canny(gray, 80, 160)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return 0.0
    idx = np.random.choice(len(xs), size=min(sample_edges, len(xs)), replace=False)
    widths = []
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        r = 9
        y0, y1 = max(0, y-r), min(gray.shape[0], y+r+1)
        x0, x1 = max(0, x-r), min(gray.shape[1], x+r+1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 5:
            continue
        hline = np.mean(patch, axis=0)
        vline = np.mean(patch, axis=1)
        for arr in (hline, vline):
            a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            p10 = np.argmax(a >= 0.1)
            p90 = np.argmax(a >= 0.9)
            if p90 > p10:
                widths.append(p90 - p10)
    return float(np.median(widths) if widths else 0.0)
