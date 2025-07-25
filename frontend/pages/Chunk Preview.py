import json
import os
import re
from typing import Any, Dict

import requests
import streamlit as st
from PIL import Image
from pathlib import Path
from collections import defaultdict

# create session_states for parameters that needs to be preserved throughout the session
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = False
if 'session_dir_chunks' not in st.session_state:
    st.session_state.session_dir_chunks = None


def load_image(img_path: str) -> Image.Image:
    return Image.open(img_path)

def list_subfolders(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def extract_x_y(filename: str):
    """
    Extract x and y from a filename like 'chunk_x0_y150.jpg'
    Returns tuple (x, y) as integers
    """
    match = re.search(r'_x(\d+)_y(\d+)', Path(filename).stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0  # fallback

def display_chunks(image_paths, max_rows=7, max_cols=7):
    """
    
    """
    if not image_paths:
        st.warning("No images to display.")
        return

    # Step 1: Group image paths by their _x value
    x_groups = defaultdict(list)
    for path in image_paths:
        x_val, y_val = extract_x_y(path)
        x_groups[x_val].append((y_val, path))

    # Step 2: Sort x keys and within each x-group, sort by y
    sorted_x_vals = sorted(x_groups.keys())
    # Limit columns to max_cols
    sorted_x_vals = sorted_x_vals[:max_cols]

    for x in sorted_x_vals:
        x_groups[x] = sorted(x_groups[x], key=lambda tup: tup[0])  # sort by y
        # Limit rows to max_rows
        x_groups[x] = x_groups[x][:max_rows]

    # Step 3: Create columns for each _x group
    st.markdown("### 🧩 Chunked Images")
    cols = st.columns(len(sorted_x_vals))

    for col_idx, x in enumerate(sorted_x_vals):
        with cols[col_idx]:
            st.markdown(f"**x = {x}**")
            for y_val, img_path in x_groups[x]:
                st.image(load_image(img_path), caption=f"y={y_val}", use_container_width=True)


st.set_page_config(page_title="Chunking Viewer", layout="wide")
st.title("Chunks Visualization")


chunk_width = chunk_height = start_pct = (
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
        "Chunking Type", ["percentage", "fixed"]
    )
    if chunking_type == "percentage":
        start_pct = st.slider("Start (%)", 5, 100, 20, 5)
        end_pct = st.slider("End (%)", start_pct, 100, 40, 5)
    elif chunking_type == "fixed":
        chunk_width = st.number_input("Chunk Width", value=640)
        chunk_height = st.number_input("Chunk Height", value=640)

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
            }
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


        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
        response = requests.post(
            "http://localhost:8000/only_chunking",
            files=files,
            data={"config": json.dumps(config)},
        )


        if response.status_code == 200:
            result = response.json()
            session_dir_chunks = result["session_path"]   
            st.session_state.session_dir_chunks = session_dir_chunks
            st.session_state.processed_chunks = True  
            st.sidebar.success("✅ Processing complete!")
        else:
            st.sidebar.error("❌ Error processing files on backend.")


# Then modify your results display section to check session state:
if st.session_state.processed_chunks and st.session_state.session_dir_chunks:

    generated_chunks_path = Path(st.session_state.session_dir_chunks) / "Generated_chunks.zip"
    with open(generated_chunks_path, "rb") as f:
        st.download_button(
            label=f"📦 Download generated chunks",
            data=f,
            file_name=f"chunks_bundle.zip",
            mime="application/zip"
        )


    # --- Results UI in main page ---

    # Step 1: Get available chunk sizes and overlaps
    image_root = os.path.join(st.session_state.session_dir_chunks, "user_images")
    imgs = os.listdir(image_root)
    selected_image = st.selectbox("Select Image", imgs)

    image_name = Path(selected_image).stem
    # path to annotated images directory, and getting list of all chunk_sizes and overlap_sizes
    chunk_root = Path(st.session_state.session_dir_chunks) / "Generated_chunks" / image_name

    pct_sizes = list_subfolders(chunk_root)
    selected_pct = st.selectbox("Select Chunk Percentage", pct_sizes)

    overlap_root = os.path.join(chunk_root, selected_pct)
    overlaps = list_subfolders(overlap_root)
    selected_overlap = st.selectbox("Select Overlap", overlaps)

    real_dir = os.path.join(overlap_root, selected_overlap)

    st.markdown("---")
    st.markdown("## 🔍 Chunked Image Pairs")

    # Load all chunk image paths
    real_imgs = [
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]

    display_chunks(real_imgs)

    
else:
    st.warning(f"⚠️ Session states not Initialized Successfully!!!!!!!!!")