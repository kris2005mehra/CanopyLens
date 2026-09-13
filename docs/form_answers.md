# Flora Carbon AI Hackathon — Official Submission Form Answers

---

### Question 1: What did you build? (3–5 sentences)
> **CanopyLens** is a deployable web application that detects individual tree crowns and measures forest canopy area from high-resolution aerial and satellite imagery (PNG, JPG, GeoTIFF) with optional KML boundary input. It integrates a pretrained RetinaNet ecological AI model (DeepForest) with spectral vegetation index analysis (Excess Green Index) to deliver individual tree counts, sub-crown foliage segmentation, and canopy cover percentages. The system adheres to a strict scientific honesty principle: real-world physical area ($m^2$ and hectares) is calculated only when reliable Ground Sampling Distance (GSD) exists or is provided, completely eliminating fabricated metrics. Users can visually inspect crown masks, toggle comparison views, and export detection coordinates and measurements to CSV.

---

### Question 2: What doesn't work? (Honest failure modes and limitations)
> 1. **Closed-Canopy Overlapping Trees:** In dense closed-canopy forests where branches intertwine seamlessly, adjacent crowns may be merged into a single detection box, leading to undercounting.
> 2. **Understory & Shadowed Vegetation:** Suppressed trees residing underneath dominant canopy layers cannot be observed in top-down 2D optical imagery. Additionally, deep solar cast shadows can cause minor crown fragmentation in the ExG mask.
> 3. **Non-Georeferenced Image + KML Spatial Overlay:** If a user supplies a standard unreferenced JPG/PNG alongside a geographic KML boundary, the system parses the KML and computes boundary area, but honestly refrains from performing an ungrounded spatial pixel-clipping overlay because coordinate registration metadata is absent.
> 4. **No Carbon or Biomass Extrapolation:** The tool does not calculate carbon stock, biomass tonnage, or CO₂ credits; doing so without LiDAR height models and species-specific allometric equations would be unscientific and misleading.

---

### Question 3: What technologies did you use?
> - **Languages & UI:** Python 3.11, Streamlit
> - **AI/ML & Vision:** PyTorch, torchvision, DeepForest (RetinaNet ResNet-50 backbone), OpenCV (`opencv-python-headless`), scikit-image, Pillow
> - **Geospatial & Vector Data:** Rasterio, Shapely, PyProj, lxml
> - **Data Analysis:** NumPy, Pandas

---

### Question 4: Did you use AI coding tools?
> **Yes**

---

### Question 5: If yes, which ones?
> - **Google Antigravity Agentic Assistant / Gemini 1.5 & Claude Opus models**
> - Used for rapid codebase scaffolding, modular directory structuring, geospatial math validation, and technical documentation drafting. All architectural decisions, package integrations, and scientific constraints were verified and tested directly.

---

### Question 6: Anything else you'd like us to know?
> CanopyLens was architected specifically around the organizer's core evaluation standard: **"A rough tool that admits what it cannot do is more valuable than a polished tool that invents figures."** Rather than relying on black-box heuristics or displaying synthetic carbon numbers, every single metric produced by CanopyLens is directly traceable to observed pixel data or verified sensor metadata. The codebase is clean, modular, and designed to be demonstrated live without developer intervention.
