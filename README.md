# CanopyLens 🌳

**AI-powered tree crown detection and canopy measurement from high-resolution forest imagery.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Model: DeepForest](https://img.shields.io/badge/Model-DeepForest_v2.1-green.svg)](https://github.com/weecology/DeepForest)

---

## 📌 Problem

High-resolution airborne and satellite imagery provides unprecedented visibility into forest ecosystems. However, assessing tree density, crown distribution, and canopy cover traditionally requires labor-intensive manual delineation or brittle, ad-hoc thresholding algorithms.

Forest conservationists, ecological researchers, and carbon monitoring initiatives need automated, reproducible tools that can:
1. Detect and count individual tree crowns.
2. Estimate the canopy area they occupy.
3. Present the findings transparently—without fabricating physical measurements or exaggerating algorithmic capabilities.

---

## 💡 Solution

**CanopyLens** is an open-source, reproducible web application that accepts high-resolution aerial forest imagery (PNG, JPG, or GeoTIFF) and optional KML boundary polygons. It runs a deep learning tree crown detection pipeline (DeepForest) combined with spectral vegetation analysis (Excess Green Index) to deliver:

- **Crown instance detection & counting**
- **Canopy pixel & physical area calculations**
- **Canopy cover percentage estimation**
- **Transparent limitation reporting** (never inventing metrics when spatial metadata is absent)
- **Data export** (CSV of crown coordinates and dimensions, annotated imagery)

---

## ✨ Features

- **Multi-Format Ingestion**: Supports standard RGB images (`.png`, `.jpg`, `.jpeg`) as well as geospatial rasters (`.tif`, `.tiff`).
- **Pretrained Ecological AI**: Utilizes DeepForest (RetinaNet with ResNet-50 backbone), trained on airborne RGB data across the National Ecological Observatory Network (NEON).
- **Sub-Crown Vegetation Masking**: Applies the Excess Green Index ($ExG = 2g - r - b$) and adaptive thresholding within detected bounding boxes to delineate foliage from non-canopy background.
- **Smart Tiling for Large Rasters**: Automatically splits high-resolution images into overlapping patches and reconciles predictions with Non-Maximum Suppression (NMS).
- **Physical Area Calculations**: Computes real-world canopy area ($m^2$ / hectares) and canopy cover percentage **only** when spatial resolution (Ground Sampling Distance / GSD) is available.
- **KML Boundary Support**: Ingests KML boundary files, extracts polygon geometries, and computes projected geographic area using local UTM projections.
- **Interactive Visualizer**: Offers multi-tab visual analytics including bounding box overlays, per-tree colored mask delineations, and side-by-side original vs. detected comparisons.
- **Data Export**: One-click download of detection metrics (CSV) and full-resolution annotated graphics.

---

## 🚀 Live Demo

- **Hosted Application**: *(Deployable to Hugging Face Spaces / Streamlit Community Cloud)*
- **Repository**: [https://github.com/kris2005mehra/CanopyLens](https://github.com/kris2005mehra/CanopyLens)

---

## 🏗️ Architecture & Pipeline

```
┌────────────────────────────────────────────────────────┐
│                      INPUT DATA                        │
│  Forest Imagery (PNG/JPG/GeoTIFF) + Optional KML       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   IMAGE VALIDATION                     │
│  - Dimension, format, and channel integrity checks     │
│  - Spatial metadata extraction (CRS, Transform, GSD)   │
│  - KML parsing & UTM projected boundary calculation   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  PREPROCESSING & TILING                │
│  - Normalization to RGB float array                    │
│  - Sliding-window tiling for large rasters (>1200px)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 DEEPFOREST INFERENCE                   │
│  - Pretrained RetinaNet (ResNet-50) detection          │
│  - Bounding box extraction with confidence scores      │
│  - Patch coordinate translation & NMS deduplication    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               VEGETATION MASK EXTRACTION               │
│  - Normalized Excess Green Index: ExG = 2g - r - b     │
│  - Adaptive thresholding inside detected crown boxes   │
│  - Pixel-accurate canopy mask & contour extraction     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  METRICS CALCULATION                   │
│  - Observed: Tree crown count                          │
│  - Calculated (pixels): Canopy pixel count, cover %    │
│  - Calculated (physical): Area in m²/ha (IF GSD valid) │
│  - STRICT HONESTY: Refuse physical area without GSD    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                 VISUALIZATION & EXPORT                 │
│  - Multi-tab overlay views, side-by-side comparison    │
│  - CSV detections export & annotated image download    │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Application & UI**: [Streamlit](https://streamlit.io/) (v1.28+)
- **Core AI Model**: [DeepForest](https://github.com/weecology/DeepForest) (v2.1+, Weecology Lab / University of Florida)
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/)
- **Computer Vision**: [OpenCV](https://opencv.org/) (`opencv-python-headless`), [scikit-image](https://scikit-image.org/), [Pillow](https://python-pillow.org/)
- **Geospatial & Vector**: [Rasterio](https://rasterio.readthedocs.io/), [Shapely](https://shapely.readthedocs.io/), [PyProj](https://pyproj4.github.io/pyproj/), [lxml](https://lxml.de/)
- **Data Manipulation**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)

---

## 🤖 Model Choice & Justification

| Candidate | Strengths | Weaknesses | Decision |
| :--- | :--- | :--- | :--- |
| **DeepForest** | Purpose-built for airborne tree crowns; trained on NEON ecological data; lightweight (~140 MB); runs fast on CPU; MIT license | Bounding boxes rather than native instance masks | **Selected as Core Detector** |
| **Detectree2** | Produces polygonal masks directly | Requires Facebook Detectron2 (fragile C++/CUDA build; frequent cloud deployment failures) | Rejected due to deployment brittleness |
| **General YOLOv8** | Fast inference | Not pretrained out-of-the-box on tree crowns; requires finding and validating custom weights | Rejected |
| **SAM (Segment Anything)** | High contour accuracy | Heavy memory footprint; high latency per prompt on CPU | ExG used for lightweight sub-crown masking |

**DeepForest** is paired with the **Excess Green Index (ExG)**. DeepForest isolates tree locations, and ExG extracts the exact canopy foliage within each box. This delivers pixel-level canopy metrics without C++ dependency issues.

---

## 📐 Canopy Area & GSD Calculation

Physical area calculations adhere to strict scientific rigor:

1. **If Ground Sampling Distance (GSD) is known**:
   $$\text{Pixel Area} = \text{GSD} \times \text{GSD} \quad (\text{m}^2)$$
   $$\text{Canopy Area} = \text{Canopy Pixels} \times \text{Pixel Area} \quad (\text{m}^2)$$
   $$\text{Canopy Cover \%} = \frac{\text{Canopy Area}}{\text{Analysis Area}} \times 100$$

2. **If GSD is unavailable**:
   The system displays:
   > *"Physical canopy area unavailable because reliable spatial resolution metadata was not provided."*
   It reports pixel counts and image-relative percentages only. It **never** manufactures physical metrics.

---

## 🗺️ KML Boundary Handling

- KML files are parsed using `lxml`.
- Polygon coordinates (latitude/longitude in WGS84, EPSG:4326) are dynamically projected to the appropriate **Universal Transverse Mercator (UTM)** zone using `pyproj`.
- Accurate geographic boundary area is computed in square meters via `shapely`.
- If an uploaded image is not georeferenced (e.g., standard JPG/PNG), the application honestly informs the user that spatial alignment cannot be guaranteed rather than performing a false overlay.

---

## ⚠️ Known Limitations

In compliance with the Flora Carbon AI evaluation criteria, the following limitations are documented:

1. **Crown Merging**: In closed-canopy or dense forest stands, adjacent interlocking crowns may be detected as a single combined crown, causing undercounting.
2. **Understory Concealment**: Suppressed understory trees hidden beneath dominant canopy layers cannot be detected in optical nadir imagery.
3. **Shadow Artifacts**: Deep topographical or solar shadows may cause tree crowns to be fragmented or excluded from the ExG vegetation mask.
4. **Geographic Generalization**: DeepForest is trained primarily on North American forest ecosystems (NEON). Tropical rainforests or arid scrublands may exhibit different morphology requiring confidence threshold adjustment.
5. **No Carbon or Biomass Fabrication**: Tree count and canopy area are **not** converted into carbon credits, CO₂ tonnage, or timber volume, as reliable biomass estimation requires species-specific allometric equations and LiDAR/field height measurements.

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Git

### 1. Clone the repository
```bash
git clone https://github.com/kris2005mehra/CanopyLens.git
cd CanopyLens
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📦 Project Structure

```
canopylens/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── packages.txt                # System packages for cloud deployment
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
│
├── src/                        # Modular processing pipeline
│   ├── __init__.py
│   ├── preprocessing.py        # Image validation, GeoTIFF CRS/GSD extraction
│   ├── inference.py            # DeepForest model caching and tiled detection
│   ├── kml_processing.py       # KML polygon parsing and UTM projection
│   ├── postprocessing.py       # ExG vegetation masking and crown extraction
│   ├── metrics.py              # Pixel and physical area metrics
│   └── visualization.py        # Crown overlays, mask contours, side-by-side
│
├── sample_data/                # Sample boundaries and test inputs
│   └── sample_boundary.kml     # Test KML ecological plot
│
└── docs/                       # Technical documentation & interview assets
    ├── methodology.md          # 2-page approach & technical design paper
    ├── form_answers.md         # Official hackathon submission answers
    └── interview_prep.md       # Technical defense & interview Q&A
```

