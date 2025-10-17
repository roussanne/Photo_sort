
# Tri Labeler – Sharp / Defocus / Motion 3-Class Tool

A Streamlit app that scores images for **Sharp**, **Defocus (out-of-focus)**, and **Motion blur**, lets you **manually correct labels** with thumbnails, and exports a dataset ready for training (`train/sharp`, `train/defocus`, `train/motion`).

https://user-images.example/tri_labeler_demo.gif

---

## ✨ Features
- **Auto scoring** using edge/frequency/directionality features
- **3-class prediction** (sharp/defocus/motion) with **live weight sliders**
- **Tile-based analysis** to catch local blur
- **Thumbnail preview** with per-image **manual relabeling**
- **CSV export** of labels & scores
- **Dataset export** to `train/` (copy or move)
- Optional **HEIC** support (iPhone photos)

---

## 📦 Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Packages listed in `requirements.txt`

```
streamlit
numpy
pandas
opencv-python
Pillow
# pillow-heif      # optional for HEIC
# scikit-image     # optional
# scipy            # optional
```

Install everything (core set):
```bash
pip install -r requirements.txt
```

HEIC support (optional):
```bash
pip install pillow-heif
```

---

## 🚀 Quick Start

1. Put `tri_labeler.py` in a folder of your choice.
2. (Optional) Edit the default folder path inside the app sidebar when it opens.
3. Run the app:
   ```bash
   streamlit run tri_labeler.py
   ```
4. Open the browser (auto-opens) at `http://localhost:8501` (or as shown in the terminal).

> **Important:** Do **not** run via `python tri_labeler.py` — use `streamlit run` (otherwise you’ll see `missing ScriptRunContext` warnings).

---

## 🧭 Using the App (Overview)

### 1) Sidebar – Folder & Scan
- **Image folder path**: Root directory to scan.
- **Include subfolders**: Recursively scan subdirectories.
- **Resize (long side)**: Normalize resolution for stable metrics (speed vs. detail trade-off).

### 2) Sidebar – Tiles & Weights
- **Tiles (N×N)**: More tiles → better at local blur, but slower.
- **Sharp weights**: VoL, Tenengrad, HighFreqRatio, EdgeSpread (inv), RadialSlope (inv).
- **Defocus weights**: EdgeSpread, VoL (inv), RadialSlope (inv), Anisotropy (inv).
- **Motion weights**: Anisotropy, StructureTensor, VoL (inv).

Use tooltips (hover over each control) for **what it means**, **when to adjust**, and **tips**.

### 3) Sidebar – Classification / Filter
- **Min scores** for each class (filter weak predictions).
- **Preview filter** to show only certain predicted class thumbnails.

### 4) Main – Thumbnails & Labeling
- Per-image **auto prediction** + **dropdown** for manual override.
- **Page size / page index** controls at the top.

### 5) Export
- **Save CSV** → writes `labels.csv` in root folder.
- **Export to train/** → copies or moves files under:
  ```
  root/
    train/
      sharp/
      defocus/
      motion/
  ```

---

## 🛠 Tips & Tuning

- Start with a **small sample (100–300 images)** to tune weights/thresholds.
- If **too many sharp images misclassified as blur**, lower defocus/motion weights or raise min sharp score.
- If **motion blur** is missed, raise **Anisotropy** / **StructureTensor** weights.
- For **bokeh-heavy portraits**, avoid pushing **EdgeSpread** too high.
- Try **Tiles = 3–5**; raise to **5–6** for telephoto/night scenes.

**Performance**: Lower the long-side resize (e.g., 896) or reduce tiles if things feel slow.

---

## ❓ Troubleshooting

**“missing ScriptRunContext” / “bare mode” warnings**  
Run with `streamlit run tri_labeler.py` (not `python` directly).

**`ModuleNotFoundError: pillow_heif`**  
Install HEIC support: `pip install pillow-heif`, or leave HEIC images out.

**OpenCV reads return `None`**  
- File is corrupted or unsupported (install `pillow-heif` for HEIC).
- Check file permissions / long path issues on Windows.

**Slow / laggy**  
- Lower **Tiles** and/or **Resize (long side)**.
- Close heavy browser tabs; prefer a Chromium-based browser.

---

## 📂 Project Layout (example)

```
your_project/
├─ tri_labeler.py
├─ requirements.txt
├─ README.md
└─ photos/                 # your images
   ├─ img001.jpg
   ├─ img002.jpg
   └─ ...
```

Output after export:
```
photos/
├─ labels.csv
└─ train/
   ├─ sharp/
   ├─ defocus/
   └─ motion/
```

---

## 🧩 Customization

- Add validation/test split on export: e.g., 80/10/10.
- Auto-normalize weights to sum to 1.
- Add histogram plots of scores for threshold selection.
- Extend with a small CNN to learn a combiner on top of features.

---

## 📄 License

This project template is provided “as is.” Add your preferred license if you plan to share or publish.




pip install -r requirements.txt


streamlit run "C:\Users\SSAFY\Desktop\Photo_sort\sort.py"

Local URL: http://localhost:8501












