import json
import logging
import math
import os
from typing import Any, Dict, List

import cv2
from Custom import stitch_chunks_custom
from model_predictions import run_model_predictions_on_chunks
from NMS import stitch_chunks_nms

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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
        chunk_dir = os.path.join(basename)
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
                    chunk_dir, f"chunk_{start_x}_{start_y}.jpg"
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
    chunk_dir = os.path.join(chunk_base_dir, basename)
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
            chunk_filename = os.path.join(chunk_dir, f"chunk_{start_x}_{start_y}.jpg")
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
        chunk_dir = os.path.join(chunk_base_dir, basename)
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
                    chunk_dir, f"chunk_{start_x}_{start_y}.jpg"
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
                basename,
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
                        chunk_dir, f"chunk_{start_x}_{start_y}.jpg"
                    )
                    cv2.imwrite(chunk_filename, chunk)
                    chunked_images.append(chunk_filename)
                    chunk_id += 1


def generate_results(session_dir: str, config_data: Dict[str, Any]) -> Dict[str, Any]:

    image_dir = os.path.join(session_dir, "images")
    chunk_base_dir = os.path.join(session_dir, "chunks")
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

    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        img_h, img_w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]

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

    # Run model predictions for all chunk
    if chunking_type == "percentage" and overlap_type in ["percentage", "dataset_pct"]:
        for dir in os.listdir(chunk_base_dir):
            dir_path = os.path.join(chunk_base_dir, dir)
            if not os.path.isdir(dir_path):
                continue

            logger.info("Traversing the directory: %s", dir_path)
            for overlap_dir in os.listdir(dir_path):
                overlap_dir_path = os.path.join(dir_path, overlap_dir)
                if not os.path.isdir(overlap_dir_path):
                    continue
                logger.info("Processing overlap directory: %s", overlap_dir_path)
                for image_dir in os.listdir(overlap_dir_path):
                    image_dir_path = os.path.join(overlap_dir_path, image_dir)
                    if not os.path.isdir(image_dir_path):
                        continue
                    logger.info("Processing image directory: %s", image_dir_path)
                    # Run model predictions on each chunk directory
                    pred_dir_list = run_model_predictions_on_chunks(image_dir_path)
    else:
        for image_dir in os.listdir(chunk_base_dir):

            image_dir_path = os.path.join(chunk_base_dir, image_dir)
            if not os.path.isdir(image_dir_path):
                continue

            # Run model predictions on each chunk directory
            pred_dir_list = run_model_predictions_on_chunks(image_dir_path)

    # Perform stitching based on the stitching type
    if stitching_type == "custom":
        min_distance_thresh = stitching_params.get("intersection_thresh", 0.5)
        comparison_thresh = stitching_params.get("comparison_thresh", 0.5)
    else:
        logger.info(f"Starting NMS stitching on {len(pred_dir_list)} directories.")
        for pred_dir in pred_dir_list:
            if not os.path.isdir(pred_dir):
                continue
            logger.info("Stitching chunks in directory: %s", pred_dir)
            stitch_chunks_nms(
                predictions_dir=pred_dir,
                image_path=image_path,
                iou_thresh=stitching_params.get("iou_thresh", 0.6),
                conf_thresh=stitching_params.get("conf_thresh", 0.3),
            )
