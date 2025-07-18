import json
import os
from typing import Any, Dict

import requests
import streamlit as st
from PIL import Image


def load_results(results_path: str) -> Dict[str, Any]:
    with open(results_path) as f:
        return json.load(f)


def load_image(img_path: str) -> Image.Image:
    return Image.open(img_path)


st.set_page_config(page_title="Chunking Viewer", layout="wide")
st.title("Chunking & Stitching Comparison")

comparison_thresh = min_distance_thresh = chunk_width = chunk_height = start_pct = (
    end_pct
) = overlap_pct = overlap_px = ovp_start = ovp_end_pct = None

# --- Sidebar: Upload and Configuration ---
with st.sidebar:
    st.header("🔧 Configuration & Upload")

    uploaded_files = st.sidebar.file_uploader(
        "📁 Upload image files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    chunking_type = st.selectbox(
        "Chunking Type", ["percentage", "fixed", "dataset-specific"]
    )
    if chunking_type == "percentage":
        start_pct = st.slider("Start (%)", 5, 100, 20, 5)
        end_pct = st.slider("End (%)", start_pct, 100, 40, 5)
    elif chunking_type == "fixed":
        chunk_width = st.number_input("Chunk Width")
        chunk_height = st.number_input("Chunk Height")
    else:
        chunk_width = st.number_input("Chunk Width", value=640, disabled=True)
        chunk_height = st.number_input("Chunk Height", value=640, disabled=True)

    overlap_type = st.selectbox(
        "Overlap Type", ["percentage", "dataset_pct", "dataset_px"]
    )
    if overlap_type == "percentage":
        ovp_start_pct = st.slider("Chunk Start (%)", 5, 100, 20, 5)
        ovp_end_pct = st.slider("Chunk End (%)", ovp_start_pct, 100, 40, 5)

    elif overlap_type == "dataset_pct":
        overlap_pct = st.number_input(
            "Overlap (%)",
            value=23,
            disabled=True,
            help="Dataset specific overlap value in %.",
        )
    else:
        overlap_px = st.number_input(
            "Overlap (px)",
            value=150,
            disabled=True,
            help="Dataset specific overlap value in pixels.",
        )

    stitching_type = st.selectbox(
        "Stitching Logic",
        ["nms", "custom"],
        help="Choose how overlapping detections are merged or resolved.",
    )

    if stitching_type == "custom":
        min_distance_thresh = st.slider(
            "Minimum Distance Threshold (px)",
            1,
            100,
            10,
            help="Minimum distance between predicted boxes to consider them separate objects.",
        )

        comparison_thresh = st.number_input(
            "Comparison Threshold (px)",
            help="Used to compare how close objects are. If the distance is less than this value, they are considered overlapping.",
        )
    else:
        iou_thresh = st.number_input(
            "IOU threshold (px)",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            help="Minimum distance between predicted boxes to consider them separate objects.",
        )

        confidence_threshold = st.number_input(
            "Confidence Threshold (px)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Used to compare how close objects are. If the distance is less than this value, they are considered overlapping.",
        )

# --- Submit Button ---
if st.sidebar.button("🚀 Submit & Process"):
    if not uploaded_files:
        st.sidebar.error("⚠️ Please upload at least one image.")
    else:
        config: Dict[str, Any] = {
            "chunking": {
                "type": chunking_type,
                "params": {},
            },
            "overlap": {
                "type": overlap_type,
                "params": {},
            },
            "stitching": {
                "type": stitching_type,
                "params": {},
            },
        }

        if chunking_type == "percentage":
            config["chunking"]["params"] = {
                "start_pct": start_pct,
                "end_pct": end_pct,
            }
        else:
            config["chunking"]["params"] = {
                "width": chunk_width,
                "height": chunk_height,
            }

        if overlap_type == "percentage":
            config["overlap"]["params"] = {
                "start_pct": ovp_start_pct,
                "end_pct": ovp_end_pct,
            }
        elif overlap_type == "dataset_pct":
            config["overlap"]["params"] = {"overlap_pct": overlap_pct}
        else:
            config["overlap"]["params"] = {"overlap_px": overlap_px}

        if stitching_type == "custom":
            config["stitching"]["params"] = {
                "intersection_thresh": min_distance_thresh,
                "comparison_thresh": comparison_thresh,
            }
        else:
            config["stitching"]["params"] = {
                "iou_thresh": iou_thresh,
                "conf_thresh": confidence_threshold,
            }

        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
        response = requests.post(
            "http://localhost:8000/upload_and_process",
            files=files,
            data={"config": json.dumps(config)},
        )

        if response.status_code == 200:
            st.sidebar.success("✅ Processing complete!")
            results_path = response.json()["results_path"]
            results = load_results(results_path)

            # --- Results UI in main page ---
            for image_name, data in results.items():
                st.markdown("---")
                st.subheader(f"📷 Image: `{image_name}`")

                st.markdown("**Original Image:**")
                st.image(load_image(data["original"]), use_column_width=True)

                # Chunk counts
                if "chunk_counts" in data:
                    st.markdown("**Chunk Counts per Combination:**")
                    st.json(data["chunk_counts"])

                # Strategy selection
                chunking_strategies = list(data["chunking_strategies"].keys())
                selected_chunking = st.selectbox(
                    "Select Chunking Strategy",
                    chunking_strategies,
                    key=f"chunking_{image_name}",
                )

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

                # Chunk Predictions
                st.markdown("### 🔍 Chunks + Predictions")
                chunk_cols = st.columns(min(5, len(selected_data["predictions"])))
                for idx, pred_img_path in enumerate(selected_data["predictions"]):
                    with chunk_cols[idx % len(chunk_cols)]:
                        st.image(
                            load_image(pred_img_path),
                            caption=f"Chunk {idx+1}",
                            use_column_width=True,
                        )

                # Stitched results
                st.markdown("### 🧵 Stitched Results")
                for stitch_type, stitched_path in selected_data["stitched"].items():
                    st.markdown(f"**{stitch_type.upper()} Stitching**")
                    if os.path.exists(stitched_path) and stitched_path.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):
                        st.image(load_image(stitched_path), use_column_width=True)
                    else:
                        st.info(f"Stitched file: {stitched_path}")

        else:
            st.sidebar.error("❌ Error processing files on backend.")
