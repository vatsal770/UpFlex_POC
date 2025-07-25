import json
import os
import re
from typing import Any, Dict

import requests
import streamlit as st
from PIL import Image
from pathlib import Path
from collections import defaultdict

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

def extract_x_y(filename: str):
    """
    Extract x and y from a filename like 'chunk_x0_y150.jpg'
    Returns tuple (x, y) as integers
    """
    match = re.search(r'_x(\d+)_y(\d+)', Path(filename).stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0  # fallback

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

    # Step 1: Group image paths by their _x value
    x_groups = defaultdict(list)
    for path in real_imgs:
        x_val, y_val = extract_x_y(path)
        x_groups[x_val].append((y_val, path))

    fixed_width = 300  # or whatever size you prefer (e.g. 250, 400, etc.)
    for i, idx in enumerate(range(start_idx, end_idx)):
        with cols[i]:
            cur_real_path = real_paths[idx]
            x_val, y_val = extract_x_y(cur_real_path)
            st.image(load_image(cur_real_path), caption=f"x={x_val}_y={y_val}", width = fixed_width)
            st.image(load_image(annotated_paths[idx]), caption=f"x={x_val}_y={y_val}", width = fixed_width)

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

    annotated_imgs = sorted([
        os.path.join(annotated_dir, f)
        for f in os.listdir(annotated_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])

    # first sort the filenames by _y and then sort it by _x on the updated image list -> this makes sure that the image chunks are visualized horizonttally first 
    # Step 1: Zip real and annotated images
    paired_images = list(zip(real_imgs, annotated_imgs))
    # Step 2: Sort the pairs based on Y first, then X from real image filename
    sorted_pairs = sorted(
        paired_images,
        key=lambda pair: (extract_x_y(os.path.basename(pair[0]))[1],  # Y value
                        extract_x_y(os.path.basename(pair[0]))[0])  # X value
    )
    # Step 3: Unzip back into separate lists
    real_imgs, annotated_imgs = zip(*sorted_pairs)
    real_imgs, annotated_imgs = list(real_imgs), list(annotated_imgs)  # Convert tuples back to lists


    if not real_imgs or not annotated_imgs:
        st.warning(f"Images not present in the designated Real and annotated chunked path")


    display_image_pairs(real_imgs, annotated_imgs)

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
