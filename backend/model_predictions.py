import json
import os
import cv2
import logging

from pathlib import Path
from typing import List
import supervision as sv
from inference import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_model_predictions_on_chunks(image_dir: str) -> List[str]:

    # Initialize the prediction model
    model = get_model(model_id="floorplan-o9hev/6", api_key="LLDV1nzXicfTYmlj9CMp")
    image_name = os.path.basename(image_dir)
    pred_dir_list = [] 

    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, image_file)
        if not os.path.isfile(image_path):
            continue

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Run inference on the image
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = bounding_box_annotator.annotate(
            scene=image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections
        )

        # Save predictions as JSON
        annotations = []
        for i in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[i]
            width = x_max - x_min
            height = y_max - y_min
            class_id = detections.class_id[i]
            confidence = float(detections.confidence[i])
            label = (
                detections.data["class_name"][i]
                if "class_name" in detections.data
                else str(class_id)
            )
            allowed_classes = ["table", "chair", "table-chair"]

            if label not in allowed_classes:
                continue

            annotation = {
                "bbox": [
                    float(x_min),
                    float(y_min),
                    float(width),
                    float(height),
                ],
                "label": label,
                "confidence": confidence,
            }
            annotations.append(annotation)

        annotation_json_dir=os.path.join(image_dir, "annotations_json")
        os.makedirs(annotation_json_dir, exist_ok=True)
        output_json_path = os.path.join(annotation_json_dir, f"{Path(image_file).stem}.json")
        with open(output_json_path, "w") as f:
            json.dump({"image": image_path, "annotations": annotations}, f, indent=4)

        annotation_img_dir=os.path.join(image_dir, "annotations_img")
        os.makedirs(annotation_img_dir, exist_ok=True)
        output_img_path = os.path.join(annotation_img_dir, f"annotated_{Path(image_file).name}")
        cv2.imwrite(output_img_path, annotated_image)
        
    pred_dir_list.append(annotation_json_dir)
    
    return pred_dir_list