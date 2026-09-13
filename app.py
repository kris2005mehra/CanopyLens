"""
CanopyLens — AI Tree Crown Detection & Canopy Analysis

A Streamlit web application for detecting individual tree crowns and estimating
canopy area from high-resolution forest imagery.

Model: DeepForest (RetinaNet/ResNet-50, pretrained on NEON airborne data)
Canopy masking: Excess Green Index (ExG) vegetation thresholding
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration (must be first Streamlit command) ---
st.set_page_config(
    page_title="CanopyLens — AI Tree Crown Analysis",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
    /* Header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4CAF50;
        margin-bottom: 0;
    }
    .tagline {
        font-size: 1.1rem;
        color: #90A4AE;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #1B2838;
        border: 1px solid #2E7D32;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetric"] label {
        color: #B0BEC5 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #4CAF50 !important;
    }

    /* Warning/info box styling */
    .limitation-box {
        background-color: #1B2838;
        border-left: 4px solid #FF9800;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .info-box {
        background-color: #1B2838;
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #607D8B;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #263238;
    }

    /* Hide Streamlit default footer */
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# --- Model Caching ---
@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the DeepForest model."""
    from src.inference import load_model
    return load_model()


# --- Header ---
st.markdown('<p class="main-title">🌳 CanopyLens</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">AI-powered tree crown detection and canopy measurement '
    "from high-resolution forest imagery</p>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Upload Data")

    uploaded_image = st.file_uploader(
        "Upload Forest Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="High-resolution aerial or satellite forest imagery. "
        "Supported formats: PNG, JPG, TIFF/GeoTIFF.",
    )

    uploaded_kml = st.file_uploader(
        "Upload KML Boundary (Optional)",
        type=["kml"],
        help="KML file defining the analysis boundary polygon. "
        "For non-georeferenced images (JPG/PNG), the KML boundary information "
        "will be displayed but cannot be spatially aligned with the image.",
    )

    st.divider()
    st.header("⚙️ Analysis Settings")

    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower = more detections (may include false positives). "
        "Higher = fewer but more confident detections.",
    )

    gsd_option = st.radio(
        "Ground Sampling Distance (GSD)",
        ["Auto-detect from image", "Enter manually", "Not available"],
        help="GSD is needed to convert pixel measurements to physical area (m²). "
        "Auto-detect works for GeoTIFF files with spatial metadata.",
    )

    manual_gsd = None
    if gsd_option == "Enter manually":
        manual_gsd = st.number_input(
            "GSD (meters per pixel)",
            min_value=0.01,
            max_value=10.0,
            value=0.5,
            step=0.01,
            format="%.2f",
        )
        st.caption("⚠️ User-provided value — not independently verified by the system.")

    st.divider()

    # Visualization options
    st.header("🎨 Visualization")
    show_boxes = st.checkbox("Show bounding boxes", value=True)
    show_masks = st.checkbox("Show canopy masks", value=True)
    show_labels = st.checkbox("Show tree IDs", value=False)

    st.divider()
    analyze_button = st.button(
        "🔍 Analyze Forest", type="primary", use_container_width=True
    )


# --- Main Content ---

# Display image preview if uploaded
if uploaded_image is not None:
    file_bytes = uploaded_image.getvalue()
    filename = uploaded_image.name

    # Show image preview
    with st.expander("📷 Image Preview", expanded=True):
        try:
            from src.preprocessing import load_image_as_rgb, prepare_for_display

            image_rgb = load_image_as_rgb(file_bytes, filename)
            preview = prepare_for_display(image_rgb, max_display_dim=1200)
            st.image(preview, caption=f"{filename} ({image_rgb.shape[1]}×{image_rgb.shape[0]} pixels)", use_container_width=True)
        except Exception as e:
            st.error(f"Could not preview image: {e}")

# Display KML info if uploaded
if uploaded_kml is not None:
    with st.expander("🗺️ KML Boundary Info", expanded=True):
        try:
            from src.kml_processing import parse_kml

            kml_bytes = uploaded_kml.getvalue()
            kml_boundary = parse_kml(kml_bytes)

            st.markdown(f"**Boundary name:** {kml_boundary.name}")
            if kml_boundary.description:
                st.markdown(f"**Description:** {kml_boundary.description}")
            st.markdown(f"**Polygons found:** {len(kml_boundary.polygons)}")

            if kml_boundary.area_sq_m is not None:
                st.markdown(f"**Polygon area:** {kml_boundary.area_sq_m:,.0f} m² "
                            f"({kml_boundary.area_hectares:.2f} hectares)")
            if kml_boundary.centroid:
                st.markdown(f"**Centroid:** {kml_boundary.centroid[1]:.4f}°N, "
                            f"{kml_boundary.centroid[0]:.4f}°E")
            if kml_boundary.crs_used:
                st.markdown(f"**Projection used:** {kml_boundary.crs_used}")

            for warning in kml_boundary.warnings:
                st.warning(warning)

            # Limitation note for non-georeferenced images
            if uploaded_image is not None:
                ext = filename.split(".")[-1].lower()
                if ext not in {"tif", "tiff"}:
                    st.info(
                        "📌 **Note:** The uploaded image is not georeferenced (JPG/PNG). "
                        "The KML boundary coordinates cannot be spatially aligned with the image. "
                        "The polygon area information above is calculated from the KML coordinates, "
                        "but analysis will be performed on the full image."
                    )

        except ValueError as e:
            st.error(f"KML parsing error: {e}")
        except Exception as e:
            st.error(f"Could not process KML file: {e}")


# --- Run Analysis ---
if analyze_button:
    if uploaded_image is None:
        st.error("Please upload a forest image to analyze.")
    else:
        file_bytes = uploaded_image.getvalue()
        filename = uploaded_image.name

        # Import all pipeline modules
        from src.preprocessing import validate_image, load_image_as_rgb, ImageMetadata
        from src.inference import detect_trees
        from src.postprocessing import (
            filter_detections,
            create_canopy_mask,
            create_crown_masks,
        )
        from src.metrics import calculate_metrics, format_area, metrics_to_dict
        from src.visualization import (
            draw_crowns,
            draw_masks_only,
            create_detection_summary_image,
        )

        # ===== STEP 1: Validate Image =====
        with st.status("Validating image...", expanded=True) as status:
            validation = validate_image(file_bytes, filename)

            if not validation.is_valid:
                for err in validation.errors:
                    st.error(f"❌ {err}")
                status.update(label="Validation failed", state="error")
                st.stop()

            metadata = validation.metadata
            st.write(f"✅ Format: {metadata.format.upper()}")
            st.write(f"✅ Dimensions: {metadata.width}×{metadata.height} pixels")
            st.write(f"✅ Channels: {metadata.channels}")
            st.write(f"✅ File size: {metadata.file_size_mb:.1f} MB")

            # GSD resolution
            gsd = None
            gsd_source = "unavailable"

            if gsd_option == "Enter manually" and manual_gsd is not None:
                gsd = manual_gsd
                gsd_source = "user_provided"
                st.write(f"📐 GSD: {gsd:.2f} m/pixel (user-provided)")
            elif metadata.gsd is not None:
                gsd = metadata.gsd
                gsd_source = metadata.gsd_source
                st.write(f"📐 GSD: {gsd:.2f} m/pixel (from GeoTIFF metadata)")
            elif gsd_option != "Not available":
                st.write("📐 GSD: Not available — physical area cannot be calculated")

            if metadata.crs:
                st.write(f"🌐 CRS: {metadata.crs}")

            for warning in validation.warnings:
                st.warning(f"⚠️ {warning}")

            status.update(label="Image validated ✓", state="complete")

        # ===== STEP 2: Load Image =====
        with st.status("Loading image...", expanded=False) as status:
            try:
                image_rgb = load_image_as_rgb(file_bytes, filename)
                st.write(f"Loaded image: {image_rgb.shape[1]}×{image_rgb.shape[0]}, "
                         f"dtype={image_rgb.dtype}")
                status.update(label="Image loaded ✓", state="complete")
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.stop()

        # ===== STEP 3: Parse KML (if provided) =====
        kml_area_m2 = None
        if uploaded_kml is not None:
            with st.status("Processing KML boundary...", expanded=False) as status:
                try:
                    from src.kml_processing import parse_kml as parse_kml_fn

                    kml_data = parse_kml_fn(uploaded_kml.getvalue())
                    if kml_data.area_sq_m:
                        kml_area_m2 = kml_data.area_sq_m
                        st.write(f"KML boundary area: {kml_area_m2:,.0f} m²")
                    status.update(label="KML processed ✓", state="complete")
                except Exception as e:
                    st.warning(f"KML processing issue: {e}. Analyzing full image.")
                    status.update(label="KML processing skipped", state="complete")

        # ===== STEP 4: Detect Trees =====
        with st.status("🌲 Detecting tree crowns...", expanded=True) as status:
            st.write("Loading DeepForest model (first run downloads ~140 MB weights)...")

            start_time = time.time()
            try:
                # Pre-load model with caching
                _ = get_model()

                detections = detect_trees(
                    image_rgb,
                    confidence_threshold=confidence,
                    patch_size=800,
                    patch_overlap=100,
                )

                elapsed = time.time() - start_time
                st.write(f"Detection complete in {elapsed:.1f}s")
                st.write(f"Raw detections: {len(detections)}")

                status.update(label=f"Detection complete — {len(detections)} crowns ✓", state="complete")

            except Exception as e:
                st.error(f"Tree detection failed: {e}")
                logger.exception("Detection error")
                status.update(label="Detection failed", state="error")
                st.stop()

        # ===== STEP 5: Post-Processing =====
        with st.status("Processing canopy masks...", expanded=False) as status:
            try:
                detections = filter_detections(
                    detections,
                    confidence_threshold=confidence,
                    min_box_pixels=100,
                )

                canopy_mask = create_canopy_mask(image_rgb, detections)
                crown_masks = create_crown_masks(image_rgb, detections)

                st.write(f"Filtered detections: {len(detections)}")
                st.write(f"Canopy pixels: {canopy_mask.sum():,}")

                status.update(label="Post-processing complete ✓", state="complete")
            except Exception as e:
                st.error(f"Post-processing failed: {e}")
                logger.exception("Post-processing error")
                st.stop()

        # ===== STEP 6: Calculate Metrics =====
        with st.status("Calculating metrics...", expanded=False) as status:
            metrics = calculate_metrics(
                detections=detections,
                canopy_mask=canopy_mask,
                image_shape=image_rgb.shape,
                crown_masks=crown_masks,
                gsd=gsd,
                gsd_source=gsd_source,
                kml_area_m2=kml_area_m2,
            )
            status.update(label="Metrics calculated ✓", state="complete")

        # ===== STEP 7: Create Visualizations =====
        with st.status("Generating visualizations...", expanded=False) as status:
            viz = create_detection_summary_image(
                image_rgb, detections, canopy_mask, crown_masks
            )

            # Custom visualization based on user settings
            custom_viz = draw_crowns(
                image_rgb,
                detections,
                canopy_mask=canopy_mask,
                crown_masks=crown_masks if show_masks else None,
                show_boxes=show_boxes,
                show_masks=show_masks,
                show_labels=show_labels,
                mask_alpha=0.35,
            )
            status.update(label="Visualizations ready ✓", state="complete")

        # ===================================================================
        # RESULTS DISPLAY
        # ===================================================================
        st.divider()
        st.header("📊 Analysis Results")

        # --- Handle no detections ---
        if len(detections) == 0:
            st.warning(
                "**No tree crowns were detected.** This does not necessarily mean no trees "
                "are present. Possible reasons:\n"
                "- Image resolution may be insufficient for individual crown detection\n"
                "- The image may not contain tree canopy visible from above\n"
                "- The confidence threshold may be set too high\n"
                "- The model may not generalize well to this type of imagery\n\n"
                "Try lowering the confidence threshold or using higher-resolution imagery."
            )
        else:
            # --- Metric Cards ---
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🌲 Trees Detected", f"{metrics.tree_count:,}")
            with col2:
                st.metric("🌿 Canopy Pixels", f"{metrics.total_canopy_pixels:,}")
            with col3:
                if metrics.canopy_area_m2 is not None:
                    if metrics.canopy_area_m2 >= 10000:
                        st.metric("📐 Canopy Area", f"{metrics.canopy_area_hectares:.2f} ha")
                    else:
                        st.metric("📐 Canopy Area", f"{metrics.canopy_area_m2:,.1f} m²")
                else:
                    st.metric("📐 Canopy Area", "N/A")
            with col4:
                if metrics.canopy_cover_percentage is not None:
                    st.metric("📊 Canopy Cover", f"{metrics.canopy_cover_percentage:.1f}%")
                else:
                    st.metric("📊 Canopy Cover", "N/A")

            # --- Spatial Resolution Context ---
            if metrics.gsd is not None:
                st.markdown(
                    f'<div class="info-box">'
                    f"<strong>Spatial resolution:</strong> {metrics.gsd:.2f} m/pixel "
                    f"({metrics.gsd_source}) — "
                    f"pixel area = {metrics.gsd * metrics.gsd:.4f} m²"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="limitation-box">'
                    "<strong>Physical canopy area unavailable</strong> — "
                    "reliable spatial resolution metadata was not provided. "
                    "Pixel-based measurements are shown instead. "
                    "To obtain area in m²/hectares, provide GSD (Ground Sampling Distance) "
                    "or use GeoTIFF imagery with embedded spatial metadata."
                    "</div>",
                    unsafe_allow_html=True,
                )

            # --- Detailed Metrics ---
            with st.expander("📋 Detailed Metrics", expanded=False):
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.markdown("**Pixel-Based Metrics**")
                    st.write(f"- Total canopy pixels: {metrics.total_canopy_pixels:,}")
                    st.write(f"- Total image pixels: {metrics.total_image_pixels:,}")
                    st.write(f"- Canopy % of image: {metrics.canopy_percentage_of_image:.1f}%")
                    st.write(f"- Mean crown area: {metrics.mean_crown_area_pixels:,.0f} pixels")
                    st.write(f"- Median crown area: {metrics.median_crown_area_pixels:,.0f} pixels")
                    if metrics.min_crown_area_pixels > 0:
                        st.write(f"- Crown area range: {metrics.min_crown_area_pixels:,} – "
                                 f"{metrics.max_crown_area_pixels:,} pixels")

                with detail_col2:
                    st.markdown("**Physical Metrics**")
                    if metrics.gsd is not None:
                        st.write(f"- GSD: {metrics.gsd:.2f} m/pixel ({metrics.gsd_source})")
                        st.write(f"- Canopy area: {format_area(metrics.canopy_area_m2)}")
                        if metrics.analysis_area_m2:
                            st.write(f"- Analysis area: {format_area(metrics.analysis_area_m2)}")
                        if metrics.canopy_cover_percentage is not None:
                            st.write(f"- Canopy cover: {metrics.canopy_cover_percentage:.1f}%")
                        if metrics.mean_crown_area_m2:
                            st.write(f"- Mean crown area: {metrics.mean_crown_area_m2:.2f} m²")
                    else:
                        st.write("Physical metrics unavailable — no GSD provided.")

                    st.markdown("**Detection Confidence**")
                    st.write(f"- Mean confidence: {metrics.mean_confidence:.3f}")
                    st.write(f"- Range: {metrics.min_confidence:.3f} – {metrics.max_confidence:.3f}")

            # --- Visualizations ---
            st.divider()
            st.header("🔍 Detection Visualization")

            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
                ["Combined View", "Crown Masks", "Bounding Boxes", "Mask Isolation"]
            )

            with viz_tab1:
                st.image(
                    custom_viz,
                    caption="Detected tree crowns (masks + bounding boxes)",
                    use_container_width=True,
                )

            with viz_tab2:
                st.image(
                    viz["masks_overlay"],
                    caption="Individual crown masks — each color represents one detected tree",
                    use_container_width=True,
                )

            with viz_tab3:
                st.image(
                    viz["boxes_overlay"],
                    caption="DeepForest bounding box detections",
                    use_container_width=True,
                )

            with viz_tab4:
                st.image(
                    viz["masks_only"],
                    caption="Canopy isolation — bright areas are detected canopy, dark areas are background",
                    use_container_width=True,
                )

            # --- Side-by-side comparison ---
            st.subheader("Original vs Detected")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                from src.preprocessing import prepare_for_display
                st.image(
                    prepare_for_display(image_rgb),
                    caption="Original Image",
                    use_container_width=True,
                )
            with comp_col2:
                st.image(
                    prepare_for_display(custom_viz),
                    caption="Detected Crowns",
                    use_container_width=True,
                )

            # --- Export ---
            st.divider()
            st.header("💾 Export Results")

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # CSV export of detections
                export_df = detections.copy()
                if metrics.gsd is not None:
                    pixel_area = metrics.gsd ** 2
                    export_df["crown_area_m2"] = (
                        (export_df["xmax"] - export_df["xmin"])
                        * (export_df["ymax"] - export_df["ymin"])
                        * pixel_area
                    )
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📄 Download Detections (CSV)",
                    csv_data,
                    file_name="canopylens_detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with export_col2:
                # Annotated image download
                annotated_pil = Image.fromarray(custom_viz)
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button(
                    "🖼️ Download Annotated Image",
                    buf.getvalue(),
                    file_name="canopylens_annotated.png",
                    mime="image/png",
                    use_container_width=True,
                )

            with export_col3:
                # Metrics summary
                metrics_dict = metrics_to_dict(metrics)
                metrics_csv = pd.DataFrame([metrics_dict]).to_csv(index=False)
                st.download_button(
                    "📊 Download Metrics Summary",
                    metrics_csv,
                    file_name="canopylens_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ===================================================================
    # LIMITATIONS & METHODOLOGY
    # ===================================================================
    st.divider()
    st.header("⚠️ Limitations & Methodology")

    st.markdown("""
**This tool provides automated estimates. Results have not been field-validated.**

The following limitations apply to all outputs:

- **Overlapping crowns** may be merged into a single detection, potentially undercounting trees in dense canopy.
- **Small or understory trees** may not be detected, especially under larger canopy.
- **Shadows and dark areas** can reduce detection accuracy.
- **Dense forests** with continuous canopy are challenging — individual tree separation degrades.
- **Image resolution** directly affects reliability. Best results with GSD ≤ 1.0 m/pixel.
- **Model generalization**: DeepForest was trained on NEON airborne data (primarily North American temperate/tropical forests). Performance on other geographies, species, or imagery types may vary.
- **Canopy masking** uses the Excess Green Index (ExG) within detected bounding boxes, which may not perfectly separate canopy from background in all conditions.
- **No species identification** is performed.
- **No biomass, carbon stock, or CO₂ sequestration estimates** are provided — these require additional models and field data that this tool does not include.
    """)

    st.markdown("""
**What this tool provides:**

| Category | Details |
|----------|---------|
| ✅ **Detected** | Individual tree crown locations (bounding boxes) |
| ✅ **Calculated** | Tree count, canopy pixel area, canopy masks |
| ✅ **Calculated (with GSD)** | Physical canopy area (m²/ha), canopy cover (%) |
| ❌ **Not provided** | Carbon stock, biomass, CO₂ sequestration, species ID |
    """)

elif uploaded_image is None:
    # Landing state — no image uploaded
    st.markdown("---")

    st.markdown("""
### How to use CanopyLens

1. **Upload** a high-resolution forest image (aerial/satellite imagery, PNG/JPG/GeoTIFF)
2. **Optionally upload** a KML file to define the analysis boundary
3. **Adjust settings** — confidence threshold and GSD if known
4. **Click "Analyze Forest"** to run the AI detection pipeline
5. **Review results** — detected crowns, tree count, canopy area, visualizations
6. **Export** detections as CSV and annotated images

### What you'll get

- Individual tree crown detection using the DeepForest AI model
- Tree count derived from detected crowns
- Canopy area estimation (in m²/hectares when spatial resolution is known)
- Visual overlay of detected crowns on your imagery
- Honest reporting of limitations and confidence levels
    """)

    st.info(
        "💡 **Tip:** For best results, use high-resolution aerial imagery with clearly visible "
        "individual tree crowns. Ground Sampling Distance (GSD) of 0.1–1.0 m/pixel works best."
    )


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "CanopyLens — Built for the Flora Carbon AI Hackathon 2025<br>"
    "Model: DeepForest (Weinstein et al., MIT License) | "
    "Canopy masking: Excess Green Index (ExG)<br>"
    "This tool provides automated estimates only. "
    "Results should be validated before use in any decision-making."
    "</div>",
    unsafe_allow_html=True,
)
