import json
import os

import requests
import streamlit as st
from PIL import Image


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


def load_image(img_path):
    return Image.open(img_path)


st.title("🧠 Chunking & Stitching Comparison Viewer (Client)")

# --- Upload Images ---
uploaded_files = st.file_uploader(
    "Upload multiple image files (Required)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# --- Configuration Form ---
with st.form(key="config_form"):
    st.markdown("### Required Chunking Configuration")

    chunking_type = st.selectbox(
        "Chunking Type", ["percentage", "fixed", "pixel", "dataset"]
    )
    overlap_type = st.selectbox("Overlap Type", ["percentage", "dataset"])
    stitching_type = st.selectbox("Stitching Logic", ["nms", "custom"])

    config = {
        "chunking_type": chunking_type,
        "overlap_type": overlap_type,
        "stitching_type": stitching_type,
    }

    if chunking_type == "percentage":
        start_pct = st.slider("Start %", min_value=5, max_value=100, step= 5, value=20)
        end_pct = st.slider("End %", min_value=start_pct, max_value=100, step = 5, value=40)
        step_pct = st.slider("Step %", min_value=5, max_value=100, step = 5, value=5)
        config.update(
            {"start_pct": start_pct, "end_pct": end_pct, "step_pct": step_pct}
        )

    elif chunking_type == "fixed":
        chunk_width = st.number_input("Chunk Width", min_value=1, value=256)
        chunk_height = st.number_input("Chunk Height", min_value=1, value=256)
        config.update({"chunk_width": chunk_width, "chunk_height": chunk_height})

    elif chunking_type == "pixel":
        black_pixel_threshold = st.number_input(
            "Black Pixel Threshold", min_value=0, value=20
        )
        intensity_threshold = st.number_input(
            "Intensity Threshold", min_value=0, max_value=255, value=10
        )
        config.update(
            {
                "black_pixel_threshold": black_pixel_threshold,
                "intensity_threshold": intensity_threshold,
            }
        )

    elif chunking_type == "dataset":
        overlap_px = st.slider("Overlap (px)", min_value=0, value=1000)
        overlap_pct = st.number_input(
            "Overlap (%)", min_value=0, max_value=100, value=10
        )
        config.update({"overlap_px": overlap_px, "overlap_pct": overlap_pct})

    if overlap_type == "percentage":
        overlap_percent = st.slider(
            "Overlap % of Chunk Size", min_value=0, max_value=100, value=30
        )
        config["overlap_percent"] = overlap_percent

    if stitching_type == "custom":
        min_distance_thresh = st.slider(
            "Minimum Distance Threshold (px)", min_value=1, value=10
        )
        config["min_distance_thresh"] = min_distance_thresh

    submitted = st.form_submit_button("Submit")

    # --- Submit to Backend ---
    if submitted:
        if not uploaded_files:
            st.error("⚠️ Please upload at least one image.")
        else:
            files = [("files", (f.name, f, f.type)) for f in uploaded_files]
            response = requests.post(
                "http://localhost:8000/upload_and_process",
                files=files,
                data={"config": json.dumps(config)},
            )

            if response.status_code == 200:
                st.success("✅ Processing complete!")
                results_path = response.json()["results_path"]
                results = load_results(results_path)

                # --- Display Results Per Image ---
                for image_name, data in results.items():
                    st.markdown("---")
                    st.subheader(f"📷 Image: `{image_name}`")

                    st.markdown("**Original Image:**")
                    st.image(load_image(data["original"]), use_column_width=True)

                    # Show chunk counts
                    if "chunk_counts" in data:
                        st.markdown("**Chunk Counts per Combination:**")
                        st.json(data["chunk_counts"])

                    # Select chunking strategy
                    chunking_strategies = list(data["chunking_strategies"].keys())
                    selected_chunking = st.selectbox(
                        "Select Chunking Strategy",
                        chunking_strategies,
                        key=f"chunking_{image_name}",
                    )

                    # Select overlap value
                    available_overlaps = list(
                        data["chunking_strategies"][selected_chunking].keys()
                    )
                    selected_overlap = st.selectbox(
                        "Select Overlap %",
                        available_overlaps,
                        key=f"overlap_{image_name}",
                    )

                    selected_data = data["chunking_strategies"][selected_chunking][
                        selected_overlap
                    ]

                    # Show predicted chunks
                    st.markdown("### 🔍 Chunks + Predictions")
                    chunk_cols = st.columns(min(5, len(selected_data["predictions"])))
                    for idx, pred_img_path in enumerate(selected_data["predictions"]):
                        with chunk_cols[idx % len(chunk_cols)]:
                            st.image(
                                load_image(pred_img_path),
                                caption=f"Chunk {idx+1}",
                                use_column_width=True,
                            )

                    # Show stitched results (show only if file exists and is an image)
                    st.markdown("### 🧵 Stitched Results")
                    for stitch_type, stitched_path in selected_data["stitched"].items():
                        st.markdown(f"**{stitch_type.upper()} Stitching**")
                        if os.path.exists(
                            stitched_path
                        ) and stitched_path.lower().endswith((".jpg", ".jpeg", ".png")):
                            st.image(load_image(stitched_path), use_column_width=True)
                        else:
                            st.info(f"Stitched file: {stitched_path}")

            else:
                st.error("❌ Error processing files on backend.")

st.markdown("---")
