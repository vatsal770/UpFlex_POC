import json
import logging
import os
import shutil

from pathlib import Path
from typing import Any, Dict, List

from chunking import ChunkProcessor
from model_predictions import ModelPredictor
from Custom import ChunkStitcher
from zip import ZipMaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# create a class for the pipeline flow
class ProcessingPipeline:
    def __init__(self, session_dir: str, config: Dict[str, Any]):
        self.session_dir = session_dir
        self.config = config

    chunker = ChunkProcessor(self.session_dir, self.config)
    backend_session_dir = chunker.generate_chunks_1()

    def run_predictions_for_all_images(self):
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
        logger.info("Starting predictions for all images in session directory: %s", self.session_dir)

        # making an instance of the class
        model_predictor = ModelPredictor(self.session_dir, self.config)

        # defining user_images directory
        image_dir = os.path.join(self.session_dir, "user_images")

        # extracting model_id, api_key, json_formats, selected_classes from the config file
        model_id = self.config["model_id"]["params"]["model_selected"]
        api_key = self.config["api_key"]["params"]["api_selected"]
        json_formats = self.config["json_formats"]["params"]["formats_selected"]
        selected_classes = self.config_data["allowed_classes"]["params"]["selected"]

        logger.info(f"Selected Classes are: {selected_classes}")    # logging selected classes


        # Extract chunking configurations
        chunking_type: str = self.config.get("chunking", {}).get("type", "")

        # Overlap configurations
        overlap_type: str = self.config.get("overlap", {}).get("type", "")

        # performing chunking and stiching per image in user_images directory
        for image_file in os.listdir(image_dir):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(image_dir, image_file)
            image_name = Path(image_path).stem

            logger.info(f"Image Path directory {image_path}")

            dataset_root = os.path.join(self.session_dir, "COCO", image_name, "Dataset")   # set COCO as a general directory for all json formats
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
                        resulting_json_directory = model_predictor.run_model_predictions_on_chunks(self.session_dir, overlap_dir_path, chunked_img_dir, full_image_name=image_name, model_id=model_id, api_key=api_key, allowed_classes=selected_classes, json_formats=json_formats)
                        pred_dir_list.append(resulting_json_directory)

            # Save entire list to JSON
            pred_info_path = os.path.join(self.session_dir, "COCO", image_name, "prediction_dir_info.json")
            with open(pred_info_path, "w") as f:
                json.dump(pred_dir_list, f, indent=2)

        logger.info("Completed predictions for all images.")


    def run_stitch_for_all_images(self):
        """
        Run stitching for all images in the session directory.

        Args:
            session_dir (str): Path to the session directory.
            config_data (Dict[str, Any]): Configuration data containing model and chunking parameters.

        Returns:
            None
        """
        logger.info("Starting stitching for all images in session directory: %s", self.session_dir)

        # creating an instance of "ChunkStitcher" class
        chunk_stitcher = ChunkStitcher(self.session_dir, self.config)

        # defining user_images directory
        image_dir = os.path.join(self.session_dir, "user_images")

        # extracting json_formats, selected_classes from the config file
        json_formats = self.config_data["json_formats"]["params"]["formats_selected"]
        selected_classes = self.config_data["allowed_classes"]["params"]["selected"]

        logger.info(f"Selected Classes are: {selected_classes}")    # logging selected classes

        # Stitching configurations
        stitching_type: str = self.config_data.get("stitching", {}).get("type", "")
        stitching_params = self.config_data.get("stitching", {}).get("params", {})

        # performing chunking and stiching per image in user_images directory
        for image_file in os.listdir(image_dir):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(image_dir, image_file)
            image_name = Path(image_path).stem

            logger.info(f"Image Path directory {image_path}")

            # define the predictions dir
            pred_dir = os.path.join(self.session_dir, "COCO", image_name, "prediction_dir_info.json")
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
                    chunk_stitcher.stitch(
                        self.session_dir,
                        predictions_dir=pred_dir,
                        image_path=image_path,
                        json_formats=json_formats,
                        merge_thresh=min_distance_thresh,
                        comparison_thresh=comparison_thresh,
                        containment_thresh=containment_thresh,
                    )


    def run_zip_all_formats(self):
        """
        Run zipping for all formats in the session directory.

        Args:
            session_dir (str): Path to the session directory.
            config_data (Dict[str, Any]): Configuration data containing model and chunking parameters.

        Returns:
            None
        """

        logger.info("Starting zipping for all formats in session directory: %s", self.session_dir)

        # create an instance of "ZipMaker" class
        zip_maker = ZipMaker(self.session_dir)

        # extracting model_id, api_key, json_formats, selected_classes from the config file
        json_formats = self.config_data["json_formats"]["params"]["formats_selected"]

        # create zipped file per json format type
        for json_format in json_formats:
            folder_name = json_format     # input_folder
            zip_output_name = folder_name + ".zip"    # output_zip_path
            zip_maker.zip_folder(folder_name, zip_output_name)
            logger.info(f"✅ Folder '{folder_name}' zipped successfully")


        
    kqjrbjf