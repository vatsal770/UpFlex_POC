import json
import os
import re
from typing import Any, Dict

import requests
import streamlit as st
from PIL import Image
from pathlib import Path

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if "json_formats" not in st.session_state:
    st.session_state.json_formats = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'session_dir' not in st.session_state:
    st.session_state.session_dir = None
if 'stiching_type' not in st.session_state:
    st.session_state.stiching_type = None



def load_image(img_path: str) -> Image.Image:
    return Image.open(img_path)

def list_subfolders(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def display_image_pairs(real_paths, annotated_paths, metadata_paths, images_per_page=4):
    """
    Display chunked as well as full images sorted by chunk_id, using metadata's chunk_id.

    Args:
        image_paths (list[str]): List of chunk image paths.
        annotated_paths (list[str]): List of annotated image paths.
        metadata_paths (list[str]): List of metadata paths for chunks.
    """
    if not real_paths or not annotated_paths:
        st.warning("No images to display.")
        return

    if len(real_paths) != len(annotated_paths):
        st.warning("Mismatch in number of real and annotated images.")
        return

    # Load metadata: mapping chunk_id -> (x, y)
    metadata_map = {}
    for meta_path in metadata_paths:
        with open(meta_path, "r") as f:
            data = json.load(f)
            chunk_id = data["chunk_id"]  # e.g., "chunk_1"
            metadata_map[chunk_id] = data

    # Sort image pairs by chunk number
    def extract_chunk_number(path):
        name = Path(path).stem  # "chunk_1"
        match = re.search(r"chunk_(\d+)", name)
        return int(match.group(1)) if match else float('inf')

    paired_images = sorted(zip(real_paths, annotated_paths), key=lambda p: extract_chunk_number(p[0]))

    # Initialize session state for pagination
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    total_pages = max(1, (len(paired_images) + images_per_page - 1) // images_per_page)

    # Navigation controls
    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        prev = st.button("← Previous", disabled=st.session_state.current_page == 0)
    with col3:
        next = st.button("Next →", disabled=st.session_state.current_page >= total_pages - 1)

    if prev:
        st.session_state.current_page -= 1
    if next:
        st.session_state.current_page += 1

    st.caption(f"Page {st.session_state.current_page + 1} of {total_pages}")

    # Display selected page
    start_idx = st.session_state.current_page * images_per_page
    end_idx = min(start_idx + images_per_page, len(paired_images))

    cols = st.columns(end_idx - start_idx)
    for i, (real_img, ann_img) in enumerate(paired_images[start_idx:end_idx]):
        with cols[i]:
            chunk_name = Path(real_img).stem  # "chunk_1"
            meta = metadata_map.get(chunk_name, {})
            x = meta.get("x", "?")
            y = meta.get("y", "?")
            st.image(load_image(real_img), caption=f"Real (x={x}, y={y})", use_container_width=True)
            st.image(load_image(ann_img), caption=f"Annotated (x={x}, y={y})", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)




st.set_page_config(page_title="Visualization", layout="wide")
st.title("Chunking & Stitching Comparison")


comparison_thresh = containment_thresh = min_distance_thresh = chunk_width = chunk_height = start_pct = (
    end_pct
) = overlap_pct = overlap_px = ovp_start = ovp_end_pct = allowed_classes = None

# --- Sidebar: Upload and Configuration ---
with st.sidebar:
    st.header("🔧 Configuration & Upload")

    uploaded_files = st.sidebar.file_uploader(
        "📁 Upload image files",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        model_id = st.selectbox(
            "Model ID",
            options=["floorplan-o9hev/6"],
            accept_new_options=True,
            index=0,  # default selection (first element)
        )

    with col2:
        api_key = st.selectbox(
            "API KEY",
            options=["LLDV1nzXicfTYmlj9CMp"],
            accept_new_options=True,
            index=0,  # default selection
        )

    json_formats = st.multiselect(
        "JSON Formats", ["COCO", "createML"],
        default=["COCO", "createML"],
    )
    # Ensure "COCO" is always included
    if "COCO" not in json_formats:
        json_formats.insert(0, "COCO")

    allowed_classes = st.multiselect(
        "Classes", ["table", "chair", "table-chair"], 
        accept_new_options=True,
        default=["table", "chair", "table-chair"],
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

    stitching_type = st.selectbox(
        "Stitching Logic",
        ["custom"],
        help="Choose how overlapping detections are merged or resolved.",
    )


    if stitching_type == "custom":
        min_distance_thresh = st.slider(
            "Minimum Distance Threshold (px)",
            1,
            100,
            12,
            help="Minimum distance between predicted boxes to consider them separate objects.",
        )

        comparison_thresh = st.number_input(
            "Comparison Threshold (px)",
            value=10,
            help="Used to compare how close objects are. If the distance is less than this value, they are considered overlapping.",
        )

        containment_thresh = st.number_input(
            "Containment Tolerance (px)",
            value=2,
            help = "Used to merge a smaller completely overlapping object into a bigger object. This variable provides tolerance values to bbox coordinates"
        )

# --- Submit Button ---
if st.sidebar.button("🚀 Submit & Process"):
    if not uploaded_files:
        st.sidebar.error("⚠️ Please upload at least one image.")
    else:
        
        st.session_state.uploaded_files = uploaded_files  # Store in session state
        st.session_state.json_formats = json_formats
        st.session_state.stiching_type = stitching_type

        config: Dict[str, Any] = {
            "model_id": {
                "params": {
                    "model_selected": model_id
                }
            },
            "api_key": {
                "params": {
                    "api_selected": api_key
                }
            },
            "json_formats": {
                "params": {
                    "formats_selected": json_formats
                }
            },
            "allowed_classes": {
                "params": {
                    "selected": allowed_classes
                },
            },
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
                "containment_thresh" : containment_thresh,
            }

        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
        response = requests.post(
            "http://localhost:8000/upload_and_process",
            files=files,
            data={"config": json.dumps(config)},
        )


        if response.status_code == 200:
            result = response.json()
            session_dir = result["session_path"]   
            st.session_state.session_dir = session_dir
            st.session_state.processed = True  
            st.sidebar.success("✅ Processing complete!")
        else:
            st.sidebar.error("❌ Error processing files on backend.")


# Then modify your results display section to check session state:
if st.session_state.processed and st.session_state.session_dir:

    # --- Results UI in main page ---
    json_formats_session = st.session_state.json_formats
    max_buttons_per_row = 5

    for row_start in range(0, len(json_formats_session), max_buttons_per_row):
        cols = st.columns(max_buttons_per_row)
        row_formats = json_formats_session[row_start:row_start + max_buttons_per_row]

        for col, json_format in zip(cols, row_formats):
            json_format_zip_path = os.path.join(
                st.session_state.session_dir, json_format + ".zip"
            )
            with col:
                with open(json_format_zip_path, "rb") as f:
                    st.download_button(
                        label=f"📦 {json_format} format",
                        data=f,
                        file_name=f"{json_format}_bundle.zip",
                        mime="application/zip"
                    )

    # Step 1: Get available chunk sizes and overlaps
    image_root = os.path.join(st.session_state.session_dir, "user_images")
    imgs = os.listdir(image_root)
    selected_image = st.selectbox("Select Image", imgs)

    image_name = Path(selected_image).stem
    # path to annotated images directory, and getting list of all chunk_sizes and overlap_sizes
    annotated_root = os.path.join(st.session_state.session_dir, image_name, "Visualize")
    dataset_root = os.path.join(st.session_state.session_dir, image_name, "Dataset", "Base_chunks")

    pct_sizes = list_subfolders(annotated_root)
    selected_pct = st.selectbox("Select Chunk Percentage", pct_sizes)

    overlap_root = os.path.join(annotated_root, selected_pct)
    overlaps = list_subfolders(overlap_root)
    selected_overlap = st.selectbox("Select Overlap", overlaps)

    overlap_dir = os.path.join(overlap_root, selected_overlap)

    st.markdown("---")
    st.markdown("## 🔍 Chunked Image Pairs")

    parts = os.path.normpath(overlap_dir).split(os.sep)
    chunk_pct, overlap = parts[-2], parts[-1]
    real_dir = os.path.join(dataset_root, chunk_pct, overlap, "chunks", "images")
    annotated_dir = os.path.join(overlap_dir, "annotated_chunks")

    real_imgs = sorted([
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    metadata_paths = sorted([
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.endswith(".json")
    ])

    annotated_imgs = sorted([
        os.path.join(annotated_dir, f)
        for f in os.listdir(annotated_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])


    if not real_imgs or not annotated_imgs:
        st.warning(f"Images not present in the designated Real and annotated chunked path")


    display_image_pairs(real_imgs, annotated_imgs, metadata_paths)

    # Step 2: Full image pair
    st.markdown("---")
    st.markdown("## 🖼 Full Image + Annotation")

    if st.session_state.stiching_type == "custom":
        full_annotations_dir = os.path.join(overlap_dir, "annotated_full_images")


    real_full = None
    real_full = os.path.join(image_root, selected_image)

    if not real_full:
        st.warning(f"⚠️ Full image not found.")

    annotated_full = None
    for f in os.listdir(full_annotations_dir):
        if f.endswith(".jpg") or f.endswith(".png"):
            annotated_full = os.path.join(full_annotations_dir, f)
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
    st.warning(f"⚠️ Session states not Initialized Successfully!!!!!!!!!")
