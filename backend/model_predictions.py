import json
import os
import cv2
import logging
import shutil

import supervision as sv
from pathlib import Path
from typing import List
from inference import get_model

# Configure logging once for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Define a class to handle model predictions
class ModelPredictor:
    def __init__(self, session_dir: str, model_id: str, api_key: str, allowed_classes: List[str], json_formats: List[str]) -> None:
        self.session_dir = session_dir
        self.model_id = model_id
        self.api_key = api_key
        self.allowed_classes = allowed_classes
        self.json_formats = json_formats

    def run_model_predictions_on_chunks(
            self,
            overlap_dir: str, 
            image_dir: str, 
            full_image_name: str,
    ) -> str:
        
        """
        Run object detection model predictions on all chunked images for a given full image.

        Parameters:
            overlap_dir (str): Path to the chunk images directory (containing overlap-specific chunks).
            image_dir (str): Path to the original full-sized input image (used for reference).
            full_image_name (str): Filename of the original input image, used to name and organize outputs.

        Returns:
            str: Path to the directory where model predictions and converted annotation files were saved.

        Functionality:
            - Iterates over all chunked images in the `overlap_dir`.
            - Sends each chunked image to the specified object detection model using the `model_id` and `api_key`.
            - Filters the model's predictions based on the `allowed_classes`.
            - Saves raw prediction results in JSON format.
            - Converts predictions to selected formats (COCO, createML, etc.) and saves them under the session directory.
            - Returns the output directory path containing predictions and annotations for downstream processing.
        """

        # category map matching labels and class_ids
        category_map = {
            "chair": 100,
            "table": 101,
            "table-chair": 103    
            }

        # Initialize the prediction model
        model = get_model(self.model_id, self.api_key)

        logger.info(f"Allowed Classes are: {self.allowed_classes}")   # logging allowed classes

        # model predictions for each chunked image
        for image_file in os.listdir(image_dir):
            if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # logger.info(f"image file {image_file}")

            image_path = os.path.join(image_dir, image_file)
            if not os.path.isfile(image_path):
                continue
            # logger.info(f"image path {image_path}")

            image_name = os.path.basename(image_path)
            stem_name = Path(image_path).stem

            metadata_path = f"{image_dir}/{stem_name}.json"
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                continue
            chunk_h, chunk_w  = image.shape[:2]
            # logger.info(f"{chunk_w}, {chunk_h}")

            # Run inference on the image
            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)

            bounding_box_annotator = sv.BoxAnnotator()

            annotated_image = bounding_box_annotator.annotate(
                scene=image, detections=detections
            )


    ############ Defining the base JSON formats ################################

            # In coco JSON format
            coco_predictions = {
                "info": {
                "year": "2024",
                "version": "14",
                "description": "Exported from roboflow.com",
                "contributor": "",
                "url": "https://public.roboflow.com/object-detection/undefined",
                "date_created": "2024-11-25T04:08:53+00:00"
                },

                "licenses": [
                    {
                        "id": 1,
                        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "name": "Public Domain"
                    }
                ],
                
                "categories": [
                    {
                        "id": 101,
                        "name": "table",
                        "supercategory": "table-chair"
                    },
                    {
                        "id": 100,
                        "name": "chair",
                        "supercategory": "table-chair"
                    },
                    {
                        "id": 103,
                        "name": "table-chair",
                        "supercategory": "none"
                    }
                ],

                "images": [
                    {
                        "id": 1, 
                        "file_name": image_name,
                        "width": chunk_w, 
                        "height":chunk_h,
                        "date_captured": "2024-11-25T04:08:53+00:00"
                    }
                ]
            }

            # In createML JSON format
            createML_predictions = [
                {
                    "image": image_name
                }
            ]
    ###############################################################



            # Create Annotations list in all json formats 
            coco_predictions["annotations"] = []
            createML_predictions[0]["annotations"] = []

            ann_id = 1
            for i in range(len(detections.xyxy)):
                x_min, y_min, x_max, y_max = detections.xyxy[i]
                width = x_max - x_min
                height = y_max - y_min
                class_id_model = detections.class_id[i]
                confidence = float(detections.confidence[i])
                label = (
                    detections.data["class_name"][i]
                    if "class_name" in detections.data
                    else str(class_id_model)
                )

                # checking for the allowed labels
                if label not in self.allowed_classes:
                    continue

                # mapping the labels back to class_ids
                cls_id = category_map[label]

                # adding coco-annotations
                coco_predictions["annotations"].append({
                    "id": ann_id,
                    "image_id": 1,
                    "category_id": cls_id,
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(width, 2),
                        round(height, 2),
                    ],
                    "area": int(width*height),
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id += 1

                if "createML" in self.json_formats:
                    # Convert to center-based coordinates
                    center_x = round(x_min + (width / 2), 2)
                    center_y = round(y_min + (height / 2), 2)

                    # adding createML-annotations
                    createML_predictions[0]["annotations"].append({
                        "label": label,
                        "coordinates": {
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height
                        }
                    })

            # Extract chunk_pct and overlap from overlap_dir
            parts = os.path.normpath(overlap_dir).split(os.sep)
            chunk_pct, overlap = parts[-2], parts[-1]


            # Dumping the annotation predictions in JSON formats
            # Dumping the coco-predictions
            annotation_json_dir_coco = os.path.join(self.session_dir, "COCO", full_image_name, "Dataset", chunk_pct, overlap, "chunks", "annotated_json_chunks")
            os.makedirs(annotation_json_dir_coco, exist_ok=True)
            output_json_path_coco = os.path.join(annotation_json_dir_coco, f"{Path(image_file).stem}.json")
            with open(output_json_path_coco, "w") as f:
                json.dump(coco_predictions, f, indent=4)

            if "createML" in self.json_formats:
                # Dumping the createML-predictions
                annotation_json_dir_createML = os.path.join(self.session_dir, "createML", full_image_name, "Dataset", chunk_pct, overlap, "chunks", "annotated_json_chunks")
                os.makedirs(annotation_json_dir_createML, exist_ok=True)
                output_json_path_createML = os.path.join(annotation_json_dir_createML, f"{Path(image_file).stem}.json")
                with open(output_json_path_createML, "w") as f:
                    json.dump(createML_predictions, f, indent=4)



            # Add the annotated images into Visualize directory
            annotation_img_dir = os.path.join(self.session_dir, "COCO", full_image_name, "Visualize", chunk_pct, overlap, "annotated_chunks")
            os.makedirs(annotation_img_dir, exist_ok=True)
            output_img_path = os.path.join(annotation_img_dir, f"annotated_{Path(image_file).name}")
            cv2.imwrite(output_img_path, annotated_image)


            # Copy the annotated chunked images into json directories
            for json_format in self.json_formats:
                if json_format == "COCO":
                    continue  # Skip copying for COCO format
                copy_chunks_img_dir = os.path.join(self.session_dir, json_format, full_image_name, "Visualize", chunk_pct, overlap, "annotated_chunks")
                os.makedirs(copy_chunks_img_dir, exist_ok=True)
                # Copy image
                shutil.copy(output_img_path, copy_chunks_img_dir)

            # Copy the chunked images into json directories
            for json_format in self.json_formats:
                if json_format == "COCO":
                    continue    # Skip copying for COCO format
                copy_chunks_img_dir = os.path.join(self.session_dir, json_format, full_image_name, "Dataset", chunk_pct, overlap, "chunks", "images")
                os.makedirs(copy_chunks_img_dir, exist_ok=True)
                # Copy image
                shutil.copy(image_path, copy_chunks_img_dir)
                shutil.copy(metadata_path, copy_chunks_img_dir)


        logger.info(f"model_predictions completed successfully!!!!!!!!!!!!!!!!!")
        # return the json_directory containing chunk annotations
        return annotation_json_dir_coco