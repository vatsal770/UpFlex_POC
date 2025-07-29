import logging
import json
import os
import cv2
import datetime

from typing import Any, Dict, List
from pathlib import Path

from zip import zip_folder    # importing zip_folder from zip.py

# Configure logging once for all modules
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def chunk_fixed_ovp_pct(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    chunk_width: int,
    chunk_height: int,
    basename: str,
    start_overlap: int,
    end_overlap: int,
    img_w: int,
    img_h: int,
    image: cv2.typing.MatLike,
): 
    
    # looping till end overlap with a step size of 5%
    for overlap_pct in range(start_overlap, end_overlap + 1, 5):
        stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))    # calculated stride (chunk_width - overlap)
        stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))    # calculated stride (chunk_height - overlap)
        # creating directory to store chunked images and metadata
        chunk_dir = os.path.join(
                chunk_base_dir,
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_pct}",
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
                    "image": basename,
                    "chunk_id": f"chunk_{chunk_id}",
                    "chunk_type": chunk_type,
                    "chunk_width": chunk_width,
                    "chunk_height": chunk_height,
                    "overlap_type": overlap_type,
                    "overlap_size": overlap_pct,
                    "x": int(start_x),
                    "y": int(start_y),
                }
                cv2.imwrite(chunk_filename, chunk)    # storing the chunked_image_file
                with open(chunk_metadata, "w") as f:    # dumping the metadata
                    json.dump(metadata, f, indent=4)
                chunk_id += 1


def chunk_fixed_ovp_data_px(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    chunk_width: int,
    chunk_height: int,
    img_w: int,
    img_h: int,
    basename: str,
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
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_px}",
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
                "image": basename,
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
            with open(chunk_metadata, "w") as f:     # dumping chunk metadata
                    json.dump(metadata, f, indent=4)
            chunk_id += 1


def chunk_pct_ovp_data_px(
    overlap_type: str,
    chunk_base_dir: str,
    chunk_type: str,
    start_pct: int,
    end_pct: int,
    img_w: int,
    img_h: int,
    basename: str,
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
                f"pct_{chunk_pct}",
                f"overlap_{overlap_px}",
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
                    "image": basename,
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
                with open(chunk_metadata, "w") as f:     # dumping chunk metadata
                    json.dump(metadata, f, indent=4)
                chunk_id += 1


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
    basename: str,
    image: cv2.typing.MatLike,
):
    # looping till end pct with a step size of 5%
    for chunk_pct in range(start_pct, end_pct + 1, 5):
        chunk_width = max(1, (img_w * chunk_pct) // 100)     # calculate chunk width
        chunk_height = max(1, (img_h * chunk_pct) // 100)    # calculate chunk height
        # looping till end overlap with a step size of 5%
        for overlap_pct in range(start_overlap, end_overlap + 1, 5):
            stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))    # calculated stride (chunk_width - overlap)
            stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))   # calculated stride (chunk_height - overlap)
            # creating directory to store chunked images and metadata
            chunk_dir = os.path.join(
                chunk_base_dir,
                f"pct_{chunk_pct}",
                f"overlap_{overlap_pct}",
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
                        "image": basename,
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
                    with open(chunk_metadata, "w") as f:     # dumping chunk metadata
                        json.dump(metadata, f, indent=4)
                    chunk_id += 1


def generate_chunks(session_dir: str, config_data: Dict[str, Any]) -> str:

    """
    Generates chunked images from the uploaded full images, based on the configuration provided.

    Parameters:
        session_dir (str): Path to the user session directory where chunks will be saved.
        config_data (Dict[str, Any]): Configuration dictionary containing chunk size, overlap strategy, etc.

    Returns:
        str: Backend-accessible path to the session directory (prepended with "./backend").

    Functionality:
        - Parses the configuration to determine how to chunk the images (e.g., percentage/fixed size).
        - Calls internal chunking logic to save the generated chunks in a structured directory.
        - Creates a zipped bundle for images and metadata and stores it in the session directory.
        - Returns a session directory path with `./backend` prepended, to be used by frontend logic.
    """

    image_dir = os.path.join(session_dir, "user_images")     # acessing user_images directory

    # Extract chunking configurations
    chunking_type: str = config_data.get("chunking", {}).get("type", "")
    chunk_params = config_data.get("chunking", {}).get("params", "")

    # Overlap configurations
    overlap_type: str = config_data.get("overlap", {}).get("type", "")
    overlap_params = config_data.get("overlap", {}).get("params", {})

    # performing chunking and stiching per image in user_images directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_path).stem

        # Created the main subfolder to be "Generated_chunks"
        chunk_base_dir = Path(session_dir) / "Generated_chunks"/ image_name
        logger.info(f"chunk directory: {chunk_base_dir}")    # check the created chunk_base_dir path

        image = cv2.imread(image_path)

        # Extracting the width and height of the image
        img_h, img_w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]  # stem name of image_file


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
                    base_name,
                    start_overlap,
                    end_overlap,
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
                    overlap_type,
                    base_name,
                    overlap_pct,
                    overlap_pct + 1,
                    img_w,
                    img_h,
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

    logger.info(f"chunks generated successfully!!!!!!!!!!!!!!!!!")

    # create zipped folder for chunked images and metadata
    folder_path = Path(session_dir) / "Generated_chunks"    # input_folder
    zip_output_path = Path(session_dir) / "Generated_chunks.zip"     # output_zip_path
    zip_folder(folder_path, zip_output_path)
    logger.info(f"✅ Folder '{folder_path}' zipped successfully at: {zip_output_path}")

    stripped_session_dir = session_dir.strip(".")
    backend_session_dir = "./backend" + stripped_session_dir   # adding "./backend" in the session_dir_path
    return backend_session_dir