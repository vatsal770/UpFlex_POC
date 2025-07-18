import os
import json
import cv2   
import re
import logging
    
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

def get_color_for_class(class_id):
    """Assign fixed RGB colors per class_id."""
    color_map = {
        100: (0, 0, 255),    # Red (chair)
        101: (0, 255, 0),    # Green (table)
        103: (255, 0, 0)     # Blue (table-chair)
    }
    return color_map.get(class_id, (128, 128, 128))  # Gray fallback

def draw_predictions_single_image(coco, image_path, output_dir):
    """
    Draw predictions from COCO-style annotations on a single image.
    
    Args:
        coco: COCO-format dictionary (images, annotations, categories)
        image_root_dir: Path to folder containing the original image
        output_dir: Path to save the annotated image
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    logger.info(f"Image loaded Successfully")
    if img is None:
        logger.info(f"⚠️ Could not load image: {image_path}")
        return

    # Map category_id to label
    id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for pred in coco["annotations"]:
        if pred["image_id"] != 1:
            continue  # skip if for some reason it's not image_id 1

        x, y, w, h = map(int, pred["bbox"])
        cat_id = pred["category_id"]
        label = id_to_name.get(cat_id, "unknown")
        color = get_color_for_class(cat_id)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} ({pred['id']})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = os.path.join(output_dir, f"annotated_Final_NMS")
        cv2.imwrite(out_path, img)
        logger.info("Image saved successfully!!!!!!!!!!!!!!!")


def stitch_chunks_nms(
    predictions_dir: str,
    image_path: str,
    iou_thresh: float=0.6,
    conf_thresh: float=0.3,
):
    """
    NMS-based stitching logic. Loads chunk predictions, maps to original image, applies NMS, and returns final predictions.
    """

    category_map = {
        "chair": 100,
        "table": 101,
        "table-chair": 102
    }

    image = cv2.imread(image_path)
    img_w, img_h = image.shape[1], image.shape[0]
    image_name = os.path.basename(image_path)

    coco_predictions = {
        "images": [
            {
                "id": 1, 
                "file_name": image_name,
                "width": img_w, 
                "height":img_h
            }
        ],
        "categories": [
            {
                "id": 101,
                "name": "table"
            },
            {
                "id": 100,
                "name": "chair"
            },
            {
                "id": 101,
                "name": "table-chair"
            }
        ]
    }

    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(predictions_dir, file)
        with open(file_path, "r") as f:
            chunk_data = json.load(f)

        chunk_image = chunk_data["image"]
        filename = Path(chunk_image.split("_x")[0] + ".jpg").stem

        annotations = chunk_data["annotations"]


        match = re.search(r"_x(\\d+)_y(\\d+)\\.jpg", filename)
        if not match:
            continue

        chunk_x = int(match.group(1))
        chunk_y = int(match.group(2))

        preds = []
        for ann in annotations:
            x1, y1, w, h = ann["bbox"]
            conf = ann["confidence"]
            label = ann["label"]
            cls_id = category_map[label]
            preds.append([x1, y1, w, h, conf, cls_id])
        all_chunk_preds.append(((chunk_x, chunk_y), preds, 1))

    # NMS logic
    category_boxes = defaultdict(list)
    category_confs = defaultdict(list)
    category_img_ids = defaultdict(list)
    for chunk_origin, preds, image_id in all_chunk_preds:
        chunk_x_min, chunk_y_min = chunk_origin
        for box_x_min, box_y_min, w, h, conf, cat in preds:
            global_box_x = int(box_x_min + chunk_x_min)
            global_box_y = int(box_y_min + chunk_y_min)
            global_box_w = min(w, img_w - global_box_x)
            global_box_h = min(h, img_h - global_box_y)
            if global_box_w <= 0 or global_box_h <= 0:
                continue

            global_box = [global_box_x, global_box_y, global_box_w, global_box_h]
            category_boxes[int(cat)].append(global_box)
            category_confs[int(cat)].append(float(conf))
            category_img_ids[int(cat)].append(image_id)

    
    coco_predictions["annotations"] = []
    ann_id = 1
    for cat_id, boxes in category_boxes.items():
        confidences = category_confs[cat_id]
        img_ids = category_img_ids[cat_id]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, iou_thresh)
        for i in indices:
            i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
            box = boxes[i]
            score = confidences[i]
            image_id = img_ids[i]
            coco_predictions.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [round(x, 2) for x in box],
                    "category_id": cat_id,
                }
            )
            ann_id += 1
    
    output_json_path = os.path.join(predictions_dir, "stitched_predictions_nms.json")
    with open(output_json_path, "w") as f:
        json.dump(coco_predictions, f, indent=4)

    draw_predictions_single_image(coco_predictions, image_path, predictions_dir)