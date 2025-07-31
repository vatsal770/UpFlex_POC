import logging
import json
import os
import cv2
import datetime 

from typing import Any, Dict, List
from pathlib import Path
from Custom import stitch_chunks_custom     # import stitch_chunks_custom from Custom.py
from model_predictions import run_model_predictions_on_chunks       # import run_model_predictions from model_predictions.py
from zip import zip_folder      # import zip_folder from zip.py

# Configure logging once for all modules
logger = logging.getLogger("chunking-1")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
console_handler.setFormatter(console_formatter)

# Prevent duplicate logs
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Apply same handler to root to catch other modules' logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not root_logger.hasHandlers():
    root_logger.addHandler(console_handler)


def chunk_fixed_ovp_pct(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type:str,
    chunk_width: int,
    chunk_height: int,
    start_overlap: int,
    end_overlap: int,
    img_w: int,
    img_h: int,
    base_name: str,
    image: cv2.typing.MatLike,
):
    
    # looping till end overlap with a step size of 5%
    for overlap_pct in range(start_overlap, end_overlap + 1, 5):
        stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))    # calculated stride (chunk_width - overlap)
        stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))   # calculated stride (chunk_height - overlap)
        # creating directory to store chunked images and metadata
        chunk_dir = os.path.join(
                chunk_base_dir,
                f"Dataset",
                f"Base_chunks",
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_pct}",
                f"chunks",
                f"images",
            )
        os.makedirs(chunk_dir, exist_ok=True)

        chunk_id = 0    # maintaining a chunk_id while generating chunks
        for y in range(0, img_h, stride_h):    # first chunking the whole image at a specific horizontal level
            chunk_end_y = min(img_h, y + chunk_height)    # checking whether the chunk_endis within the image size or not
            start_y = max(0, chunk_end_y - chunk_height)    # if not, then we start our last vertical chunk from (chunk_end_y - chunk_height). No padding involved
            for x in range(0, img_w, stride_w):      # then chunking the image at a specific vertical level
                chunk_end_x = min(img_w, x + chunk_width)    # checking whether the chunk_endis within the image size or not
                start_x = max(0, chunk_end_x - chunk_width)    # if not, then we start our last horizontal chunk from (chunk_end_x - chunk_width). No padding involved
                chunk = image[
                    int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
                ]
                # generating the chunked image_name to be chunk_{chunk_id}.jpg
                chunk_filename = os.path.join(
                    chunk_dir, f"chunk_{chunk_id}.jpg"
                )
                # generating the chunk metadata as chunk_{chunk_id}.json
                chunk_metadata = os.path.join(chunk_dir, f"chunk_{chunk_id}.json")

                metadata = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "image": base_name,
                    "chunk_id": f"chunk_{chunk_id}",
                    "chunk_type": chunk_type,
                    "chunk_width": chunk_width,
                    "chunk_height": chunk_height,
                    "overlap_type": overlap_type,
                    "overlap_size": overlap_pct,
                    "x": int(start_x),
                    "y": int(start_y),
                }

                cv2.imwrite(chunk_filename, chunk)   # storing the chunked_image_file
                with open(chunk_metadata, "w") as f:     # dump chunk metadata
                    json.dump(metadata, f, indent=4)
                chunk_id += 1

                if chunk_end_x == img_w:
                    break    # if chunk_end_x is equal to img_w, then we break the loop

            if chunk_end_y == img_h:
                break    # if chunk_end_y is equal to img_h, then we break the loop


def chunk_fixed_ovp_data_px(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    chunk_width: int,
    chunk_height: int,
    img_w: int,
    img_h: int,
    base_name: str,
    overlap_px: int,
    image: cv2.typing.MatLike,
):
    logger.info(
        f"Chunking with fixed size {chunk_width}x{chunk_height} and overlap {overlap_px}px"
    )
    stride_w = stride_h = int(chunk_width - overlap_px)
    # creating directory to store chunked images and metadata
    chunk_dir = os.path.join(
                chunk_base_dir,
                f"Dataset",
                f"Base_chunks",
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_px}",
                f"chunks",
                f"images",
            )
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_id = 0    # maintaining a chunk_id while generating chunks
    for y in range(0, img_h, stride_h):    # first chunking the whole image at a specific horizontal level
        chunk_end_y = min(img_h, y + chunk_height)    # checking whether the chunk_endis within the image size or not
        start_y = max(0, chunk_end_y - chunk_height)    # if not, then we start our last vertical chunk from (chunk_end_y - chunk_height). No padding involved
        for x in range(0, img_w, stride_w):      # then chunking the image at a specific vertical level
            chunk_end_x = min(img_w, x + chunk_width)    # checking whether the chunk_endis within the image size or not
            start_x = max(0, chunk_end_x - chunk_width)    # if not, then we start our last horizontal chunk from (chunk_end_x - chunk_width). No padding involved
            chunk = image[
                int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
            ]
            chunk_filename = os.path.join(chunk_dir, f"chunk_{chunk_id}.jpg")    # generating the chunked image_name to be chunk_{chunk_id}.jpg
            chunk_metadata = os.path.join(chunk_dir, f"chunk_{chunk_id}.json")    # generating the chunk metadata as chunk_{chunk_id}.json
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "image": base_name,
                "chunk_id": f"chunk_{chunk_id}",
                "chunk_type": chunk_type,
                "chunk_width": chunk_width,
                "chunk_height": chunk_height,
                "overlap_type": overlap_type,
                "overlap_size": overlap_px,
                "x": int(start_x),
                "y": int(start_y),
            }
            cv2.imwrite(chunk_filename, chunk)   # storing the chunked_image_file
            with open(chunk_metadata, "w") as f:     # dump chunk metadata
                json.dump(metadata, f, indent=4)
            chunk_id += 1

            if chunk_end_x == img_w:
                break    # if chunk_end_x is equal to img_w, then we break the loop

        if chunk_end_y == img_h:
            break    # if chunk_end_y is equal to img_h, then we break the loop

def chunk_pct_ovp_data_px(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    start_pct: int,
    end_pct: int,
    img_w: int,
    img_h: int,
    base_name: str,
    overlap_px: int,
    image: cv2.typing.MatLike,
):
    # looping till end pct with a step size of 5%
    for chunk_pct in range(start_pct, end_pct + 1, 5):
        chunk_width = max(1, (img_w * chunk_pct) // 100)     # calculate chunk width
        chunk_height = max(1, (img_h * chunk_pct) // 100)    # calculate chunk height

        stride_w = stride_h = chunk_width - overlap_px
        # creating directory to store chunked images and metadata
        chunk_dir = os.path.join(
                chunk_base_dir,
                f"Dataset",
                f"Base_chunks",
                f"pct_{chunk_pct}",
                f"overlap_{overlap_px}",
                f"chunks",
                f"images",
            )
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_id = 0    # maintaining a chunk_id while generating chunks
        for y in range(0, img_h, stride_h):    # first chunking the whole image at a specific horizontal level
            chunk_end_y = min(img_h, y + chunk_height)    # checking whether the chunk_endis within the image size or not
            start_y = max(0, chunk_end_y - chunk_height)    # if not, then we start our last vertical chunk from (chunk_end_y - chunk_height). No padding involved
            for x in range(0, img_w, stride_w):      # then chunking the image at a specific vertical level
                chunk_end_x = min(img_w, x + chunk_width)    # checking whether the chunk_endis within the image size or not
                start_x = max(0, chunk_end_x - chunk_width)    # if not, then we start our last horizontal chunk from (chunk_end_x - chunk_width). No padding involved
                chunk = image[
                    int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
                ]
                chunk_filename = os.path.join(chunk_dir, f"chunk_{chunk_id}.jpg")    # generating the chunked image_name to be chunk_{chunk_id}.jpg
                chunk_metadata = os.path.join(chunk_dir, f"chunk_{chunk_id}.json")    # generating the chunk metadata as chunk_{chunk_id}.json
                metadata = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "image": base_name,
                    "chunk_id": f"chunk_{chunk_id}",
                    "chunk_type": chunk_type,
                    "chunk_width": chunk_width,
                    "chunk_height": chunk_height,
                    "overlap_type": overlap_type,
                    "overlap_size": overlap_px,
                    "x": int(start_x),
                    "y": int(start_y),
                }

                cv2.imwrite(chunk_filename, chunk)   # storing the chunked_image_file
                with open(chunk_metadata, "w") as f:     # dump chunk metadata 
                    json.dump(metadata, f, indent=4)
                chunk_id += 1

                if chunk_end_x == img_w:
                    break    # if chunk_end_x is equal to img_w, then we break the loop

            if chunk_end_y == img_h:
                break    # if chunk_end_y is equal to img_h, then we break the loop


def chunk_pct_ovp_pct(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    start_pct: int,
    end_pct: int,
    img_w: int,
    img_h: int,
    start_overlap: int,
    end_overlap: int,
    base_name: str,
    image: cv2.typing.MatLike,
):
    # looping till end pct with a step size of 5%
    for chunk_pct in range(start_pct, end_pct + 1, 5):
        chunk_width = max(1, (img_w * chunk_pct) // 100)    # calculate chunk width
        chunk_height = max(1, (img_h * chunk_pct) // 100)   # calculate chunk height
        # looping till end overlap with a step size of 5%
        for overlap_pct in range(start_overlap, end_overlap + 1, 5):
            stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))   # calculated stride (chunk_width - overlap)
            stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))  # calculated stride (chunk_height - overlap)
            # creating directory to store chunked images and metadata
            chunk_dir = os.path.join(
                chunk_base_dir,
                f"Dataset",
                f"Base_chunks",
                f"pct_{chunk_pct}",
                f"overlap_{overlap_pct}",
                f"chunks",
                f"images",
            )
            os.makedirs(chunk_dir, exist_ok=True)

            chunk_id = 0    # maintaining a chunk_id while generating chunks
            for y in range(0, img_h, stride_h):    # first chunking the whole image at a specific horizontal level
                chunk_end_y = min(img_h, y + chunk_height)    # checking whether the chunk_endis within the image size or not
                start_y = max(0, chunk_end_y - chunk_height)    # if not, then we start our last vertical chunk from (chunk_end_y - chunk_height). No padding involved
                for x in range(0, img_w, stride_w):      # then chunking the image at a specific vertical level
                    chunk_end_x = min(img_w, x + chunk_width)    # checking whether the chunk_endis within the image size or not
                    start_x = max(0, chunk_end_x - chunk_width)    # if not, then we start our last horizontal chunk from (chunk_end_x - chunk_width). No padding involved
                    chunk = image[
                        int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
                    ]
                    chunk_filename = os.path.join(chunk_dir, f"chunk_{chunk_id}.jpg")    # generating the chunked image_name to be chunk_{chunk_id}.jpg
                    chunk_metadata = os.path.join(chunk_dir, f"chunk_{chunk_id}.json")    # generating the chunk metadata as chunk_{chunk_id}.json

                    metadata = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "image": base_name,
                        "chunk_id": f"chunk_{chunk_id}",
                        "chunk_type": chunk_type,
                        "chunk_width": chunk_width,
                        "chunk_height": chunk_height,
                        "overlap_type": overlap_type,
                        "overlap_size": overlap_pct,
                        "x": int(start_x),
                        "y": int(start_y),
                    }

                    cv2.imwrite(chunk_filename, chunk)  # storing the chunked_image_file
                    with open(chunk_metadata, "w") as f:    # dump chunk metadata
                        json.dump(metadata, f, indent=4)
                    chunk_id += 1

                    if chunk_end_x == img_w:
                        break    # if chunk_end_x is equal to img_w, then we break the loop

                if chunk_end_y == img_h:
                    break    # if chunk_end_y is equal to img_h, then we break the loop


def generate_results(session_dir: str, config_data: Dict[str, Any]) -> str:

    """
    Full implementaion for chunking, model predictions, stitching, and packaging results.

    Parameters:
        session_dir (str): Path to the user session directory for input/output operations.
        config_data (Dict[str, Any]): Configuration dictionary containing all required parameters for processing.

    Returns:
        str: Backend-accessible path to the session directory (prepended with "./backend").

    Functionality:
        - Generates chunked images based on the given configuration.
        - Runs model predictions on each chunk and saves the raw prediction files.
        - Applies stitching logic to merge per-chunk predictions into full-image annotations.
        - Converts the merged results into selected JSON formats (COCO, createML, etc.).
        - Creates a zipped bundle for each JSON format and stores it in the session directory.
        - Returns a session directory path with `./backend` prepended, to be used by frontend logic.
    """

    # defining user_images directory
    image_dir = os.path.join(session_dir, "user_images")

    # extracting model_id, api_key, json_formats, selected_classes from the config file
    model_id = config_data["model_id"]["params"]["model_selected"]
    api_key = config_data["api_key"]["params"]["api_selected"]
    json_formats = config_data["json_formats"]["params"]["formats_selected"]
    selected_classes = config_data["allowed_classes"]["params"]["selected"]

    logger.info(f"Selected Classes are: {selected_classes}")    # logging selected classes


    # Extract chunking configurations
    chunking_type: str = config_data.get("chunking", {}).get("type", "")
    chunk_params = config_data.get("chunking", {}).get("params", "")

    # Overlap configurations
    overlap_type: str = config_data.get("overlap", {}).get("type", "")
    overlap_params = config_data.get("overlap", {}).get("params", {})

    # Stitching configurations
    stitching_type: str = config_data.get("stitching", {}).get("type", "")
    stitching_params = config_data.get("stitching", {}).get("params", {})

    # performing chunking and stiching per image in user_images directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_path).stem

        # Changed the main subfolders to be image_names
        chunk_base_dir = os.path.join(session_dir, image_name)

        logger.info(f"Image Path directory {image_path}")
        image = cv2.imread(image_path)

        # extracting image width and height
        img_h, img_w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]

        # checking for each possible combinations of chunking
        if chunking_type == "percentage":
            start_pct = chunk_params.get("start_pct", 0)
            end_pct = chunk_params.get("end_pct", 0)

            # chunking: percentage, overlap:percentage
            if overlap_type == "percentage":
                start_overlap = overlap_params.get("start_pct", 0)
                end_overlap = overlap_params.get("end_pct", 0)
                chunk_pct_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    start_pct,
                    end_pct,
                    img_w,
                    img_h,
                    start_overlap,
                    end_overlap,
                    base_name,
                    image,
                )

            # chunking: percentage, overlap:dataset_pct
            elif overlap_type == "dataset_pct":
                overlap_pct = (
                    config_data.get("overlap", {})
                    .get("params", "")
                    .get("overlap_pct", "")
                )
                chunk_pct_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    start_pct,
                    end_pct,
                    img_w,
                    img_h,
                    overlap_pct,
                    overlap_pct + 1,
                    base_name,
                    image,
                )

            # chunking: percentage, overlap:dataset_px
            else:
                overlap_px = overlap_params.get("overlap_px", 0)
                chunk_pct_ovp_data_px(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    start_pct,
                    end_pct,
                    img_w,
                    img_h,
                    base_name,
                    overlap_px,
                    image,
                )

        elif chunking_type == "fixed":
            chunk_width = chunk_params.get("width", 640)
            chunk_height = chunk_params.get("height", 640)
            
            # chunking: fixed, overlap:percentage
            if overlap_type == "percentage":
                start_overlap = overlap_params.get("start_pct", 0)
                end_overlap = overlap_params.get("end_pct", 0)
                chunk_fixed_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    chunk_width,
                    chunk_height,
                    start_overlap,
                    end_overlap,
                    base_name,
                    img_w,
                    img_h,
                    image,
                )

            # chunking: fixed, overlap:dataset_pct
            elif overlap_type == "dataset_pct":
                overlap_pct = (
                    config_data.get("overlap", {})
                    .get("params", "")
                    .get("overlap_pct", "")
                )
                chunk_fixed_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    chunk_width,
                    chunk_height,
                    overlap_pct,
                    overlap_pct + 1,
                    img_w,
                    img_h,
                    base_name,
                    image,
                )

            # chunking: fixed, overlap:dataset_px
            else:
                overlap_px = overlap_params.get("overlap_px", 0)
                chunk_fixed_ovp_data_px(
                    overlap_type,
                    chunk_base_dir,
                    chunking_type,
                    chunk_width,
                    chunk_height,
                    img_w,
                    img_h,
                    base_name,
                    overlap_px,
                    image,
                )


        dataset_root = os.path.join(chunk_base_dir, "Dataset", "Base_chunks")   # general directory for all json formats
        # Run model predictions for all chunk
        if chunking_type in ["percentage", "fixed"] and overlap_type in ["percentage", "dataset_pct", "dataset_px"]:
            pred_dir_list = []  # to store directories with json prediction files
            for dir in os.listdir(dataset_root):
                dir_path = os.path.join(dataset_root, dir)
                if not os.path.isdir(dir_path):
                    continue

                logger.info("Traversing the directory: %s", dir_path)
                for overlap_dir in os.listdir(dir_path):
                    overlap_dir_path = os.path.join(dir_path, overlap_dir)
                    if not os.path.isdir(overlap_dir_path):
                        continue
                    logger.info("Processing overlap directory: %s", overlap_dir_path)

                    chunked_img_dir = os.path.join(overlap_dir_path, "chunks", "images")    # directory containing chunked images and metadata

                    # Run model predictions on each chunk directory
                    resulting_json_directory = run_model_predictions_on_chunks(session_dir, overlap_dir_path, chunked_img_dir, full_image_name=image_name, model_id=model_id, api_key=api_key, allowed_classes=selected_classes, json_formats=json_formats)
                    pred_dir_list.append(resulting_json_directory)


        # Perform stitching based on the stitching type
        if stitching_type == "custom":
            logger.info(f"Starting Custom stitching on {len(pred_dir_list)} directories.")

            # Extract stitching parameters
            min_distance_thresh = stitching_params.get("intersection_thresh", 0.5)
            comparison_thresh = stitching_params.get("comparison_thresh", 0.5)
            containment_thresh = stitching_params.get("containment_thresh", 0.5)

            for pred_dir in pred_dir_list:
                if not os.path.isdir(pred_dir):
                    continue

                logger.info("Stiching chunks in directory: %s", {pred_dir})
                stitch_chunks_custom(
                    session_dir,
                    predictions_dir=pred_dir,
                    image_path=image_path,
                    json_formats=json_formats,
                    merge_thresh=min_distance_thresh,
                    comparison_thresh=comparison_thresh,
                    containment_thresh=containment_thresh,
                )

    # create zipped file per json format type
    for json_format in json_formats:
        folder_path = os.path.join(session_dir, json_format)     # input_folder
        zip_output_path = folder_path + ".zip"    # output_zip_path
        zip_folder(folder_path, zip_output_path)
        logger.info(f"✅ Folder '{folder_path}' zipped successfully at: {zip_output_path}")

    stripped_session_dir = session_dir.strip(".")
    backend_session_dir = "./backend" + stripped_session_dir    # adding "./backend" in the session_dir_path
    return backend_session_dir
