import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_color_for_class(class_id):
    """Assign fixed RGB colors per class_id."""
    color_map = {
        100: (0, 0, 255),  # Red (chair)
        101: (0, 255, 0),  # Green (table)
        103: (255, 0, 0),  # Blue (table-chair)
    }
    return color_map.get(class_id, (128, 128, 128))  # Gray fallback


def draw_predictions_single_image(coco, image_path, output_dir):
    """
    Draw predictions from COCO-style annotations on a single image with image_id=1.

    Args:
        coco: COCO-format dictionary (images, annotations, categories)
        image_path: Path to the original image
        output_dir: Path to save the annotated image
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"⚠️ Could not load image: {image_path}")
        return

    logger.info("✅ Image loaded successfully")

    # Build category_id → name map
    id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    drawn_labels = set()

    for pred in coco["annotations"]:
        if pred["image_id"] != 1:
            continue

        x, y, w, h = map(int, pred["bbox"])
        category_id = pred["category_id"]
        label = id_to_name.get(category_id, "unknown")
        color = get_color_for_class(category_id)

        # Draw rectangle and prediction ID
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, str(pred["id"]), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Only write label once per category
        if label not in drawn_labels:
            cv2.putText(img, label, (10, 30 + 25 * len(drawn_labels)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            drawn_labels.add(label)

    out_path = os.path.join(output_dir, "annotated_Final_NMS.jpg")
    cv2.imwrite(out_path, img)
    logger.info(f"✅ Saved annotated image to: {out_path}")


def stitch_chunks_nms(
    predictions_dir: str,
    image_path: str,
    iou_thresh: float = 0.6,
    conf_thresh: float = 0.3,
):
    """
    NMS-based stitching logic. Loads chunk predictions, maps to original image, applies NMS, and returns final predictions.
    """
    logger.info(f"Starting NMS stitching.")

    category_map = {"chair": 100, "table": 101, "table-chair": 103}

    image = cv2.imread(image_path)
    img_w, img_h = image.shape[1], image.shape[0]
    image_name = os.path.basename(image_path)

    logger.info(f"Image dimensions: {img_w}, {img_h}")

    coco_predictions = {
        "images": [{"id": 1, "file_name": image_name, "width": img_w, "height": img_h}],
        "categories": [
            {"id": 101, "name": "table"},
            {"id": 100, "name": "chair"},
            {"id": 103, "name": "table-chair"},
        ],
    }

    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue

        logger.info("Loading the annotation file.")

        file_path = os.path.join(predictions_dir, file)
        with open(file_path, "r") as f:
            chunk_data = json.load(f)

        chunk_image = chunk_data["image"]
        filename = Path(chunk_image).stem

        annotations = chunk_data["annotations"]

        logger.info("Successfully laoded the annotation file.")
        match = re.search(r"_x(\d+)_y(\d+)", filename)
        logger.info(f"Match performed on {filename}. results: {match}")

        if not match:
            continue

        chunk_x = int(match.group(1))
        chunk_y = int(match.group(2))

        logger.info(f"Processing chunk at ({chunk_x}, {chunk_y})")
        logger.info(f"Processing the json files to extract all the annotations. {annotations}")
        preds = []
        for ann in annotations:
            x1, y1, w, h = ann["bbox"]
            conf = ann["confidence"]
            label = ann["label"]
            cls_id = category_map[label]
            preds.append([x1, y1, w, h, conf, cls_id])
        all_chunk_preds.append(((chunk_x, chunk_y), preds, 1))

    logger.info(f"Applying NMS logic on {len(all_chunk_preds)} chunks.")

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

    logger.info(f"Applying NMS logic on orignal image scaled backed bboxes, {len(category_boxes)} chunks.")
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
            coco_predictions["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [round(x, 2) for x in box],
                    "category_id": cat_id,
                }
            )
            ann_id += 1

    logger.info(f"Total predictions after NMS: {len(coco_predictions['annotations'])}")
    stitched_dir = os.path.join(predictions_dir,"stitched_predictions_nms")
    os.makedirs(stitched_dir, exist_ok=True)
    output_json_path = os.path.join(stitched_dir, "stitched_predictions_nms.json")
    with open(output_json_path, "w") as f:
        json.dump(coco_predictions, f, indent=4)

    logger.info(f"NMS stitching completed, json saved at {output_json_path}. Drawing predictions on the image.")
    draw_predictions_single_image(coco_predictions, image_path, stitched_dir)
