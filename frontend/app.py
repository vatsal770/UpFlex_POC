import json
import os
from typing import Any, Dict

import requests
import streamlit as st
from PIL import Image
from pathlib import Path

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'session_dir' not in st.session_state:
    st.session_state.session_dir = None


def load_results(results_path: str) -> Dict[str, Any]:
    with open(results_path) as f:
        return json.load(f)


def load_image(img_path: str) -> Image.Image:
    return Image.open(img_path)

def list_subfolders(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])


def display_image_pairs(real_paths, annotated_paths, images_per_page=4):
    if not real_paths or not annotated_paths:
        st.warning("No images to display.")
        return

    if len(real_paths) != len(annotated_paths):
        st.warning("Mismatch in number of real and annotated images.")
        return

    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    total_pages = max(1, (len(real_paths) + images_per_page - 1) // images_per_page)

    # Navigation controls
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        prev = st.button("← Previous", disabled=st.session_state.current_page == 0)
    with col3:
        next = st.button("Next →", disabled=st.session_state.current_page >= total_pages - 1)

    if prev:
        st.session_state.current_page = max(0, st.session_state.current_page - 1)

    if next:
        st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)

    st.caption(f"Page {st.session_state.current_page + 1} of {total_pages}")

    # Calculate indices for current page
    start_idx = st.session_state.current_page * images_per_page
    end_idx = min((st.session_state.current_page + 1) * images_per_page, len(real_paths))
    
    # Ensure we always have at least 1 column
    num_columns = max(1, end_idx - start_idx)

    # Horizontal scrolling container
    st.markdown("""
    <style>
        .scrolling-wrapper {
            overflow-x: auto;
            display: flex;
            flex-wrap: nowrap;
        }
        .scrolling-wrapper > div {
            flex: 0 0 auto;
            margin-right: 20px;
        }
    </style>
    <div class="scrolling-wrapper">
    """, unsafe_allow_html=True)

    # Create columns for current images
    cols = st.columns(num_columns)

    fixed_width = 300  # or whatever size you prefer (e.g. 250, 400, etc.)
    for i, idx in enumerate(range(start_idx, end_idx)):
        with cols[i]:
            st.image(load_image(real_paths[idx]), caption="Real", width = fixed_width)
            st.image(load_image(annotated_paths[idx]), caption="Annotated", width = fixed_width)

    st.markdown("</div>", unsafe_allow_html=True)



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
        st.session_state.uploaded_files = uploaded_files  # Store in session state
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
        else:
            st.sidebar.error("❌ Error processing files on backend.")


session_dir = "/home/vatsal/Documents/analysis/backend/uploads/Visualization"
st.session_state.session_dir = session_dir
st.session_state.processed = True

# Then modify your results display section to check session state:
if st.session_state.processed and st.session_state.session_dir:

    # --- Results UI in main page ---

    # Step 1: Get available chunk sizes and overlaps
    chunks_root = os.path.join(session_dir, "chunks")
    pct_sizes = list_subfolders(chunks_root)
    selected_pct = st.selectbox("Select Chunk Percentage", pct_sizes)

    overlap_dir = os.path.join(chunks_root, selected_pct)
    overlaps = list_subfolders(overlap_dir)
    selected_overlap = st.selectbox("Select Overlap", overlaps)

    image_root = os.path.join(overlap_dir, selected_overlap)

    st.markdown("---")
    st.markdown("## 🔍 Chunked Image Pairs")

    images_dir = os.path.join(session_dir, "images")
    image_path = os.listdir(images_dir)[0]
    image_name = Path(image_path).stem

    real_dir = os.path.join(image_root, image_name)
    annotated_dir = os.path.join(real_dir, "annotations_img")

    if not os.path.exists(annotated_dir):
        st.warning(f"No annotated folder for image: {image_name}")

    real_imgs = sorted([
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    annotated_imgs = sorted([
        os.path.join(annotated_dir, f)
        for f in os.listdir(annotated_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    if not real_imgs or not annotated_imgs:
        st.warning(f"Images not present in the designated Real and annotated chunked path")

    display_image_pairs(real_imgs, annotated_imgs)

    # Step 2: Full image pair
    st.markdown("---")
    st.markdown("## 🖼 Full Image + Annotation")

    images_dir = os.path.join(session_dir, "images")
    annotations_dir = os.path.join(session_dir, "chunks", selected_pct, selected_overlap, image_name, "annotations_json")

    real_full = None
    for f in os.listdir(images_dir):
        if f.endswith(".jpg") or f.endswith(".png"):
            real_full = os.path.join(images_dir, f)
            break

    if not real_full:
        st.warning(f"⚠️ Full image not found.")

    annotated_full = None
    for f in os.listdir(annotations_dir):
        if f.endswith(".jpg") or f.endswith(".png"):
            annotated_full = os.path.join(annotations_dir, f)
            break

    if not annotated_full:
        st.warning(f"⚠️ annotated_full image not found.")

    if real_full and annotated_full:
        cols = st.columns(2)
        with cols[0]:
            st.image(load_image(real_full), caption="Full Image", use_container_width=True)
        with cols[1]:
            st.image(load_image(annotated_full), caption="Annotated Full Image", use_container_width=True)
    else:
        st.warning("⚠️ Full or annotated full image not found.")

else:
    st.warning(f"Session states not Initialized Successfully!!!!!!!!!")
