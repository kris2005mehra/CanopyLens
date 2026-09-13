# CanopyLens — In-Person Technical Interview & Demo Preparation

This guide prepares you to walk the judges through **CanopyLens** during an in-person technical interview at the Flora Carbon AI Kolkata office.

---

## 1. The 60-Second Demo Walkthrough Flow

1. **The Hook (15s):**  
   *"CanopyLens is an AI tool for forestry and environmental teams to detect individual tree crowns, count trees, and accurately measure canopy area without fabricating physical metrics."*
2. **Input Upload (15s):**  
   - Open the web interface.
   - Upload a sample forest image (PNG/JPG or GeoTIFF).
   - Point out the validation status (dimensions, format, GSD resolution check).
   - (Optional) Upload `sample_boundary.kml` to show how KML parsing extracts the boundary and computes UTM geodesic area.
3. **Execute Analysis (15s):**  
   - Click **"Analyze Forest"**.
   - Explain the live steps: Tiling -> DeepForest detection -> ExG sub-crown vegetation masking -> Metrics calculation.
4. **Present Results & Integrity (15s):**  
   - Highlight the 4 metric cards: Trees Detected, Canopy Pixels, Canopy Area, Canopy Cover %.
   - **Show the honesty feature:** *"Notice that if GSD is absent, the system displays canopy pixels and clearly states physical area cannot be computed, rather than inventing square meters."*
   - Toggle between **Combined View**, **Crown Masks**, and **Side-by-Side Comparison**.
   - Show the CSV export with per-crown areas.

---

## 2. Likely Technical Interview Questions & Defensible Answers

### Q1: Why did you choose DeepForest over Detectree2 or YOLO?
> **Answer:**  
> "Detectree2 uses Facebook's Detectron2 framework, which relies on custom C++/CUDA extensions. In cloud deployment environments like Hugging Face Spaces or Streamlit Cloud, Detectron2 frequently triggers compilation timeouts and wheel mismatches. Standard YOLO models are trained on COCO (everyday objects), so they require extensive fine-tuning for aerial forestry. DeepForest is purpose-built by ecological researchers at the University of Florida (Weecology Lab), pre-trained on airborne data across 22 NEON ecological sites, installs cleanly with standard PyTorch wheels, and runs fast on CPU (~0.8s per tile)."

---

### Q2: DeepForest produces bounding boxes. How did you get canopy area?
> **Answer:**  
> "Bounding boxes overestimate tree canopy by 20% to 45% because rectangles include background soil, shadows, and corner gaps. To solve this without adding a heavy model like SAM, we implemented the **Normalized Excess Green Index** ($ExG = 2g - r - b$) combined with adaptive Otsu thresholding inside each detected bounding box. This creates pixel-accurate masks for the foliage of each tree, allowing us to compute genuine canopy pixel area."

---

### Q3: How do you calculate real-world physical area ($m^2$ and hectares)?
> **Answer:**  
> "We use the Ground Sampling Distance (GSD), which represents the real-world distance between pixel centers (e.g., 0.5 m/pixel).
> - $\text{Pixel Area} = \text{GSD} \times \text{GSD} = 0.5 \times 0.5 = 0.25 \text{ m}^2$
> - $\text{Canopy Area} = \text{Canopy Pixels} \times \text{Pixel Area}$
> - If the user uploads a GeoTIFF, we extract GSD directly from the affine transform and CRS. If it's a JPG/PNG, the user can manually specify GSD. If GSD is not available, we strictly refuse to calculate square meters and report pixel counts only."

---

### Q4: How do you handle KML boundaries?
> **Answer:**  
> "KML boundaries are parsed with `lxml`. Because raw coordinates are in degrees (WGS84, EPSG:4326), calculating area directly in degrees is mathematically invalid. We compute the polygon's centroid, identify the appropriate local **Universal Transverse Mercator (UTM)** projection zone (e.g., EPSG:32613), project the polygon using `pyproj`, and compute the geodesic area in square meters with `shapely`."

---

### Q5: How do you handle overlapping or interlocking tree crowns?
> **Answer:**  
> "In dense closed-canopy stands, overlapping crowns are a known challenge for optical nadir imagery. DeepForest uses Non-Maximum Suppression (NMS) with an IoU threshold of 0.3. When crowns strongly overlap, the model may merge them into a single detection box. We explicitly display a warning in the UI: *'Adjacent or overlapping crowns may be merged, which can result in undercounting.'*"

---

### Q6: Why didn't you calculate carbon stock or CO₂ tonnage?
> **Answer:**  
> "Because that would be fabricating data. Estimating carbon stock requires knowing the wood volume and biomass density, which depends on tree height (usually measured with LiDAR or photogrammetric point clouds), diameter at breast height (DBH), and species-specific allometric equations. Optical 2D satellite imagery alone does not provide height or species. Calling 2D crown area 'carbon stock' without those parameters is unscientific."

---

### Q7: What would you improve if given 2 more weeks?
> **Answer:**  
> "1. **Multi-spectral & LiDAR Integration:** Incorporate near-infrared (NDVI) bands and LiDAR Canopy Height Models (CHMs) for 3D canopy volume and accurate understory separation.  
> 2. **MobileSAM fine-tuning:** Integrate lightweight MobileSAM for edge-case crown polygon boundaries in complex tropical forests.  
> 3. **Automated Orthomosaic Tiling Service:** Implement asynchronous background tile workers for gigabyte-scale GeoTIFFs using Celery or Redis queues."
