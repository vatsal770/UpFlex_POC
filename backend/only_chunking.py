import logging
import os
import cv2

from typing import Any, Dict, List
from pathlib import Path

from zip import zip_folder

# Configure logging once for all modules
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def chunk_fixed_ovp_pct(
    chunk_base_dir: str,
    chunk_width: int,
    chunk_height: int,
    overlap_type: str,
    basename: str,
    start_overlap: int,
    end_overlap: int,
    img_w: int,
    img_h: int,
    image: cv2.typing.MatLike,
):
    for overlap_pct in range(start_overlap, end_overlap + 1, 5):
        stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))
        stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))
        chunk_dir = os.path.join(
                chunk_base_dir,
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_pct}",
            )
        os.makedirs(chunk_dir, exist_ok=True)

        chunked_images: List[str] = []
        chunk_id = 0
        for y in range(0, img_h, stride_h):
            chunk_end_y = min(img_h, y + chunk_height)
            start_y = max(0, chunk_end_y - chunk_height)
            for x in range(0, img_w, stride_w):
                chunk_end_x = min(img_w, x + chunk_width)
                start_x = max(0, chunk_end_x - chunk_width)
                chunk = image[
                    int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
                ]
                chunk_filename = os.path.join(
                    chunk_dir, f"chunk_x{int(start_x)}_y{int(start_y)}.jpg"
                )
                cv2.imwrite(chunk_filename, chunk)
                chunked_images.append(chunk_filename)
                chunk_id += 1


def chunk_fixed_ovp_data_px(
    chunk_base_dir: str,
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
    chunk_dir = os.path.join(
                chunk_base_dir,
                f"size_{chunk_width}-{chunk_height}",
                f"overlap_{overlap_px}",
            )
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_id = 0
    logger.info(f"{img_h}, {stride_h}, {stride_w}")
    for y in range(0, img_h, stride_h):
        chunk_end_y = min(img_h, y + chunk_height)
        start_y = max(0, chunk_end_y - chunk_height)
        for x in range(0, img_w, stride_w):
            chunk_end_x = min(img_w, x + chunk_width)
            start_x = max(0, chunk_end_x - chunk_width)
            chunk = image[
                int(start_y) : int(chunk_end_y), int(start_x) : int(chunk_end_x)
            ]
            chunk_filename = os.path.join(chunk_dir, f"chunk_x{int(start_x)}_y{int(start_y)}.jpg")
            cv2.imwrite(chunk_filename, chunk)
            chunk_id += 1


def chunk_pct_ovp_data_px(
    chunk_base_dir: str,
    start_pct: int,
    end_pct: int,
    img_w: int,
    img_h: int,
    basename: str,
    overlap_px: int,
    image: cv2.typing.MatLike,
):

    for chunk_pct in range(start_pct, end_pct + 1, 5):
        chunk_width = max(1, (img_w * chunk_pct) // 100)
        chunk_height = max(1, (img_h * chunk_pct) // 100)

        stride_w = stride_h = chunk_width - overlap_px
        chunk_dir = os.path.join(
                chunk_base_dir,
                f"pct_{chunk_pct}",
                f"overlap_{overlap_px}",
            )
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_id = 0
        for y in range(0, img_h, stride_h):
            chunk_end_y = min(img_h, y + chunk_height)
            start_y = max(0, chunk_end_y - chunk_height)
            for x in range(0, img_w, stride_w):
                chunk_end_x = min(img_w, x + chunk_width)
                start_x = max(0, chunk_end_x - chunk_width)
                chunk = image[start_y:chunk_end_y, start_x:chunk_end_x]
                chunk_filename = os.path.join(
                    chunk_dir, f"chunk_x{int(start_x)}_y{int(start_y)}.jpg"
                )
                cv2.imwrite(chunk_filename, chunk)
                chunk_id += 1


def chunk_pct_ovp_pct(
    overlap_type: str,
    chunk_base_dir: str,
    start_pct: int,
    end_pct: int,
    img_w: int,
    img_h: int,
    start_overlap: int,
    end_overlap: int,
    basename: str,
    image: cv2.typing.MatLike,
):

    for chunk_pct in range(start_pct, end_pct + 1, 5):
        chunk_width = max(1, (img_w * chunk_pct) // 100)
        chunk_height = max(1, (img_h * chunk_pct) // 100)

        for overlap_pct in range(start_overlap, end_overlap + 1, 5):
            stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))
            stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))
            chunk_dir = os.path.join(
                chunk_base_dir,
                f"pct_{chunk_pct}",
                f"overlap_{overlap_pct}",
            )
            os.makedirs(chunk_dir, exist_ok=True)

            chunked_images: List[str] = []
            chunk_id = 0
            for y in range(0, img_h, stride_h):
                chunk_end_y = min(img_h, y + chunk_height)
                start_y = max(0, chunk_end_y - chunk_height)
                for x in range(0, img_w, stride_w):
                    chunk_end_x = min(img_w, x + chunk_width)
                    start_x = max(0, chunk_end_x - chunk_width)
                    chunk = image[start_y:chunk_end_y, start_x:chunk_end_x]
                    chunk_filename = os.path.join(
                        chunk_dir, f"chunk_x{int(start_x)}_y{int(start_y)}.jpg"
                    )
                    cv2.imwrite(chunk_filename, chunk)
                    chunked_images.append(chunk_filename)
                    chunk_id += 1


def generate_chunks(session_dir: str, config_data: Dict[str, Any]) -> Dict[str, Any]:

    image_dir = os.path.join(session_dir, "user_images")
    image_path = pred_dir_list = None
    results: Dict[str, Any] = {}

    # Extract chunking configurations
    chunking_type: str = config_data.get("chunking", {}).get("type", "")
    chunk_params = config_data.get("chunking", {}).get("params", "")

    # Overlap configurations
    overlap_type: str = config_data.get("overlap", {}).get("type", "")
    overlap_params = config_data.get("overlap", {}).get("params", {})

    # Stitching configurations
    stitching_type: str = config_data.get("stitching", {}).get("type", "")
    stitching_params = config_data.get("stitching", {}).get("params", {})

    # performing chunking and stiching for each image in user_images directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_path).stem

        # Created the main subfolders to be "Generated_chunks"
        chunk_base_dir = Path(session_dir) / "Generated_chunks"/ image_name
        logger.info(f"chunk directory: {chunk_base_dir}")

        logger.info(f"Image Path directory {image_path}")
        image = cv2.imread(image_path)

        # Extracting the width and height of the image
        img_h, img_w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]


        # checking for each possible combinations of chunking
        if chunking_type == "percentage":
            start_pct = chunk_params.get("start_pct", 0)
            end_pct = chunk_params.get("end_pct", 0)
            if overlap_type == "percentage":
                start_overlap = overlap_params.get("start_pct", 0)
                end_overlap = overlap_params.get("end_pct", 0)
                chunk_pct_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    start_pct,
                    end_pct,
                    img_w,
                    img_h,
                    start_overlap,
                    end_overlap,
                    base_name,
                    image,
                )
            elif overlap_type == "dataset_pct":
                overlap_pct = (
                    config_data.get("overlap", {})
                    .get("params", "")
                    .get("overlap_pct", "")
                )
                chunk_pct_ovp_pct(
                    overlap_type,
                    chunk_base_dir,
                    start_pct,
                    end_pct,
                    img_w,
                    img_h,
                    overlap_pct,
                    overlap_pct + 1,
                    base_name,
                    image,
                )
            else:
                overlap_px = overlap_params.get("overlap_px", 0)
                chunk_pct_ovp_data_px(
                    chunk_base_dir,
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

            if overlap_type == "percentage":
                start_overlap = overlap_params.get("start_pct", 0)
                end_overlap = overlap_params.get("end_pct", 0)
                chunk_fixed_ovp_pct(
                    chunk_base_dir,
                    chunk_width,
                    chunk_height,
                    overlap_type,
                    base_name,
                    start_overlap,
                    end_overlap,
                    img_w,
                    img_h,
                    image,
                )
            elif overlap_type == "dataset_pct":
                overlap_pct = (
                    config_data.get("overlap", {})
                    .get("params", "")
                    .get("overlap_pct", "")
                )
                chunk_fixed_ovp_pct(
                    chunk_base_dir,
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
            else:
                overlap_px = overlap_params.get("overlap_px", 0)
                chunk_fixed_ovp_data_px(
                    chunk_base_dir,
                    chunk_width,
                    chunk_height,
                    img_w,
                    img_h,
                    base_name,
                    overlap_px,
                    image,
                )

    logger.info(f"chunks generated successfully!!!!!!!!!!!!!!!!!")
    folder_path = Path(session_dir) / "Generated_chunks"    # input_folder
    zip_output_path = Path(session_dir) / "Generated_chunks.zip"     # output_zip_path
    zip_folder(folder_path, zip_output_path)
    logger.info(f"✅ Folder '{folder_path}' zipped successfully at: {zip_output_path}")

    stripped_session_dir = session_dir.strip(".")
    backend_session_dir = "./backend" + stripped_session_dir   # adding "./backend" in the session_dir_path
    return backend_session_dir