import json
import logging
import os

from pathlib import Path
from typing import Any, Dict, List

from chunking import ChunkProcessor
from model_predictions import ModelPredictor
from Custom import ChunkStitcher
from zip import ZipMaker

# Configure logging once for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# create a class for the pipeline flow
class ProcessingPipeline:
    def __init__(self, session_dir: str, model_id: str, api_key: str, selected_classes: str, json_formats: List[str], chunking_type: str, chunk_params: Dict[str, Any], overlap_type: str, overlap_params: Dict[str, Any], stitching_type: str, stitching_params: Dict[str, Any]):
        self.session_dir = session_dir
        self.model_id = model_id
        self.api_key = api_key
        self.selected_classes = selected_classes
        self.json_formats = json_formats
        self.chunking_type = chunking_type
        self.chunk_params = chunk_params
        self.overlap_type = overlap_type
        self.overlap_params = overlap_params
        self.stitching_type = stitching_type
        self.stitching_params = stitching_params

    def run_chunking_for_all_images(self) -> None:
        chunker = ChunkProcessor(self.session_dir, self.chunking_type, self.chunk_params, self.overlap_type, self.overlap_params)
        chunker.generate_chunks_1()

    def run_predictions_for_all_images(self) -> None:
        """
        Run model predictions for all chunked images in the session directory.

        Returns:
            None
        """
        logger.info("Starting predictions for all images in session directory: %s", self.session_dir)

        # making an instance of the class
        model_predictor = ModelPredictor(self.session_dir, self.model_id, self.api_key, self.selected_classes, self.json_formats)

        # defining user_images directory
        image_dir = os.path.join(self.session_dir, "user_images")

        logger.info(f"Selected Classes are: {self.selected_classes}")    # logging selected classes

        # performing chunking and stiching per image in user_images directory
        for image_file in os.listdir(image_dir):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            image_path = os.path.join(image_dir, image_file)
            image_name = Path(image_path).stem

            logger.info(f"Image Path directory {image_path}")

            dataset_root = os.path.join(self.session_dir, "COCO", image_name, "Dataset")   # set COCO as a general directory for all json formats
            # Run model predictions for all chunk
            if self.chunking_type in ["percentage", "fixed"] and self.overlap_type in ["percentage", "dataset_pct", "dataset_px"]:
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
                        resulting_json_directory = model_predictor.run_model_predictions_on_chunks(overlap_dir_path, chunked_img_dir, full_image_name=image_name)
                        pred_dir_list.append(resulting_json_directory)

            # Save entire list to JSON
            pred_info_path = os.path.join(self.session_dir, "COCO", image_name, "prediction_dir_info.json")
            with open(pred_info_path, "w") as f:
                json.dump(pred_dir_list, f, indent=2)

        logger.info("Completed predictions for all images.")


    def run_stitch_for_all_images(self) -> None:
        """
        Run stitching for all images in the session directory.

        Returns:
            None
        """
        logger.info("Starting stitching for all images in session directory: %s", self.session_dir)

        # creating an instance of "ChunkStitcher" class
        chunk_stitcher = ChunkStitcher(self.session_dir, self.json_formats, self.stitching_type, self.stitching_params)

        # defining user_images directory
        image_dir = os.path.join(self.session_dir, "user_images")

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
            if self.stitching_type == "custom":
                logger.info(f"Starting Custom stitching on {len(pred_dir_list)} directories.")

                for pred_dir in pred_dir_list:
                    if not os.path.isdir(pred_dir):
                        continue

                    logger.info("Stiching chunks in directory: %s", {pred_dir})
                    chunk_stitcher.stitch(
                        predictions_dir=pred_dir,
                        image_path=image_path,
                    )


    def run_zip_all_formats(self) -> None:
        """
        Run zipping for all formats in the session directory.

        Returns:
            None
        """

        logger.info("Starting zipping for all formats in session directory")

        # create an instance of "ZipMaker" class
        zip_maker = ZipMaker(self.session_dir)

        # create zipped file per json format type
        for json_format in self.json_formats:
            folder_name = json_format     # input_folder
            zip_output_name = folder_name + ".zip"    # output_zip_path
            zip_maker.zip_folder(folder_name, zip_output_name)
            logger.info(f"✅ Folder '{folder_name}' zipped successfully")


    