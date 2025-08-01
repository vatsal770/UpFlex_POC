import logging
import os
import json

from typing import Any, Dict
from pathlib import Path

from model_predictions import run_model_predictions_on_chunks       # import run_model_predictions from model_predictions.py



# Configure logging once for all modules
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_predictions_for_all_images(
    session_dir: str, 
    config_data: Dict[str, Any]
):

    """
    Run model predictions for all images in the session directory.

    Args:
        session_dir (str): Path to the session directory.
        model_id (str): Model ID to use for predictions.
        api_key (str): API key for authentication.
        allowed_classes (List[str]): List of allowed classes for predictions.
        json_formats (List[str]): List of JSON formats to generate.

    Returns:
        None
    """
    logger.info("Starting predictions for all images in session directory: %s", session_dir)

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

    # Overlap configurations
    overlap_type: str = config_data.get("overlap", {}).get("type", "")

    # performing chunking and stiching per image in user_images directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_path).stem

        logger.info(f"Image Path directory {image_path}")

        dataset_root = os.path.join(session_dir, "COCO", image_name, "Dataset")   # set COCO as a general directory for all json formats
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

        # Save entire list to JSON
        pred_info_path = os.path.join(session_dir, "COCO", image_name, "prediction_dir_info.json")
        with open(pred_info_path, "w") as f:
            json.dump(pred_dir_list, f, indent=2)

    logger.info("Completed predictions for all images.")