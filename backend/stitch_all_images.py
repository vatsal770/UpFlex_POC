import logging
import json
import os

from typing import Any, Dict
from pathlib import Path
from Custom import stitch_chunks_custom     # import stitch_chunks_custom from Custom.py

# Configure logging once for all modules
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_stitch_for_all_images(
    session_dir: str, 
    config_data: Dict[str, Any]
):
    """
    Run stitching for all images in the session directory.

    Args:
        session_dir (str): Path to the session directory.
        config_data (Dict[str, Any]): Configuration data containing model and chunking parameters.

    Returns:
        None
    """
    logger.info("Starting stitching for all images in session directory: %s", session_dir)

    # defining user_images directory
    image_dir = os.path.join(session_dir, "user_images")

    # extracting json_formats, selected_classes from the config file
    json_formats = config_data["json_formats"]["params"]["formats_selected"]
    selected_classes = config_data["allowed_classes"]["params"]["selected"]

    logger.info(f"Selected Classes are: {selected_classes}")    # logging selected classes

    # Stitching configurations
    stitching_type: str = config_data.get("stitching", {}).get("type", "")
    stitching_params = config_data.get("stitching", {}).get("params", {})

    # performing chunking and stiching per image in user_images directory
    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_path).stem

        logger.info(f"Image Path directory {image_path}")

        # define the predictions dir
        pred_dir = os.path.join(session_dir, "COCO", image_name, "prediction_dir_info.json")
        # extract the prediction directories from the predictions JSON file
        with open(pred_dir, "r") as f:
            pred_dir_list = json.load(f)

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
    
