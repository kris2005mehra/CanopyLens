# CanopyLens — AI Tree Crown & Canopy Analysis
**Maximum 2-Page Technical Methodology & Architecture Document**  
*Flora Carbon AI Solo Hiring Hackathon Submission*  
*Author:* Kris Mehra  
*Target Role:* Senior Full-Stack + Geospatial AI Engineer  

---

## 1. APPROACH

Assessing forest canopy density and individual tree crown distribution from high-resolution airborne or satellite optical imagery is a foundational task in computational forestry and carbon credit verification. Conventional approaches either rely on manual visual tagging (non-scalable) or uncalibrated color thresholding (prone to false positives from understory grasses and soil). 

**CanopyLens** solves this by establishing a two-stage hybrid computer vision and remote sensing pipeline:
1. **Object Detection Stage:** We employ **DeepForest** (a specialized RetinaNet with a ResNet-50 backbone), trained on airborne imagery from the National Ecological Observatory Network (NEON), to isolate discrete tree crown regions.
2. **Sub-Crown Delineation Stage:** Rather than treating rectangular bounding boxes as solid canopy, we compute the **Normalized Excess Green Index** ($ExG = 2g - r - b$) with localized adaptive thresholding inside each detection box to isolate genuine foliage pixels from inter-canopy gaps and shadows.
3. **Physical Conversion Integrity:** Strict physical dimensional constraints are enforced: real-world metrics ($m^2$, hectares, canopy cover %) are calculated **only** when true spatial resolution (Ground Sampling Distance / GSD) is available. If spatial resolution is unknown, the application reports pixel metrics and explicitly warns that physical area is not calculable.

---

## 2. ARCHITECTURE & WORKFLOW

```
High-Res Imagery (PNG/JPG/GeoTIFF) + Optional KML
   │
   ▼
[ 1. Image & Metadata Validation ]
   ├── Dimensions, channel integrity, file size checks
   ├── Spatial resolution check: GeoTIFF transform/CRS extraction OR user-declared GSD
   └── KML polygon parsing & UTM projection area calculation (EPSG:326XX / EPSG:327XX)
   │
   ▼
[ 2. Preprocessing & Tiling Engine ]
   ├── Dynamic patch division (800×800 px with 100 px overlap) for large orthomosaics
   └── Edge padding to prevent border artifacting
   │
   ▼
[ 3. DeepForest AI Inference ]
   ├── RetinaNet ResNet-50 feature extraction & anchor box classification
   └── Coordinate mapping to global image space + Non-Maximum Suppression (IoU: 0.3)
   │
   ▼
[ 4. Post-Processing & Vegetation Delineation ]
   ├── Excess Green Index ($ExG$) computed across detection regions
   ├── Adaptive Otsu thresholding to segment leaf clusters from ground/shadow
   └── Per-crown instance mask extraction & contour extraction
   │
   ▼
[ 5. Canopy Metrics Calculation ]
   ├── Observed: Tree count ($N$ detected crown instances)
   ├── Pixel Metrics: Total canopy pixels, mean/median crown pixel area
   └── Physical Metrics: Canopy Area ($m^2$, ha) and Cover % (ONLY if GSD verified)
   │
   ▼
[ 6. Interactive Visualization & Export ]
   ├── Layered overlays (Boxes, Masks, Contours, Mask Isolation)
   └── Downstream export: CSV detections table & annotated high-res images
```

---

## 3. KEY TECHNICAL DECISIONS

### A. Model Selection: DeepForest vs. Detectree2 vs. YOLO
- **Detectree2 (Mask R-CNN):** While Detectree2 natively outputs instance masks, it requires Facebook's `detectron2` library, which depends on custom C++/CUDA extensions. In cloud environments (Streamlit Cloud, Hugging Face Spaces), this frequently causes build timeouts, GCC mismatch errors, or runtime compilation failures.
- **YOLOv8-seg:** Excellent raw speed, but standard weights are trained on MS-COCO (potted plants, humans, cars). Fine-tuning on diverse forest crown datasets for a weekend hackathon introduces high risk of hallucinated predictions.
- **DeepForest (Chosen):** Purpose-built for forestry remote sensing; pre-trained across 22 NEON ecological sites covering diverse canopy structures; installs cleanly via pure PyTorch wheels; runs efficiently on CPU (~0.8s per patch); MIT licensed.

### B. Sub-Crown Foliage Delineation: Excess Green Index (ExG)
Bounding boxes overestimate tree canopy area by 20%–45% because they include non-canopy corner pixels, ground shadows, and bare soil. By computing the normalized Excess Green Index ($ExG = 2g - r - b$) within each detected box and applying Otsu thresholding, we convert rectangular boxes into tight foliage masks without introducing complex, slow segmentation backbones like SAM.

### C. Geospatial Rigor in KML & Area Computations
Computing polygon areas using raw geographic coordinates (WGS84 degrees) yields nonsensical figures. CanopyLens automatically computes the centroid of KML boundaries, determines the appropriate local **Universal Transverse Mercator (UTM)** zone, projects the polygon via `pyproj`, and calculates geodesic area in square meters.

---

## 4. WHAT WORKED

- **Individual Crown Detection:** DeepForest demonstrated strong detection capability on open and semi-open forest canopies, correctly isolating dominant and co-dominant crowns.
- **Patch Tiling & NMS:** Tiling large imagery with 100px overlap and running torchvision Non-Maximum Suppression seamlessly handled large scenes without memory exhaustion.
- **Zero-Fabrication Policy:** The system gracefully handles unreferenced imagery by withholding physical measurements and explaining why, adhering strictly to the hackathon's honesty principle.
- **Export Pipeline:** Direct export of detection tables with spatial coordinates and crown areas provides immediate value for downstream GIS workflows (QGIS, ArcGIS).

---

## 5. WHAT DIDN'T WORK & LIMITATIONS

- **Closed-Canopy Overlapping Crowns:** In dense, continuous tropical rainforests with overlapping interlocking crowns, the model occasionally groups multi-tree clusters into single large bounding boxes (crown merging undercounting).
- **Suppressed Understory Trees:** Optical nadir imagery cannot penetrate dominant tree canopy; understory seedlings and sub-canopy vegetation are inevitably missed.
- **Deep Shadows:** Strong solar zenith angles cast dark shadows on tree flanks, occasionally causing the ExG threshold to clip darker foliage.
- **Non-Georeferenced KML Overlay:** When users upload non-georeferenced JPG/PNG images with a KML, automated pixel-to-geographic coordinate registration is impossible without ground control points. The app deliberately declines false spatial overlays and clearly notifies the user.

---

## 6. SCIENTIFIC & CARBON BOUNDARIES

CanopyLens explicitly distinguishes between **observed data** (tree locations), **calculated metrics** (tree count, canopy area, canopy cover %), and **unsupported claims** (carbon stock, biomass, CO₂ sequestration). Estimating carbon stock from 2D optical imagery without tree height (LiDAR/Stereo-photogrammetry) and species-specific allometric equations is scientifically invalid; CanopyLens refuses to manufacture such claims.
