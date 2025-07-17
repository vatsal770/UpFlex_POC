import os
import json
import cv2   
import re
    
from collections import defaultdict
import numpy as np


def stitch_chunks_nms(
    predictions_dir,
    original_sizes,
    image_name_to_id,
    categories_map,
    iou_thresh=0.6,
    conf_thresh=0.3,
):
    """
    NMS-based stitching logic. Loads chunk predictions, maps to original image, applies NMS, and returns final predictions.
    """

    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(predictions_dir, file)
        with open(file_path, "r") as f:
            chunk_data = json.load(f)
        chunk_path = chunk_data["image"]
        annotations = chunk_data["annotations"]
        filename = os.path.basename(chunk_path)

        match = re.search(r"_x(\\d+)_y(\\d+)\\.jpg", filename)
        if not match:
            continue

        chunk_x = int(match.group(1))
        chunk_y = int(match.group(2))
        original_image_name = filename.split("_x")[0] + ".jpg"
        image_id = image_name_to_id[original_image_name]
        original_size = original_sizes[original_image_name]

        preds = []
        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = ann["confidence"]
            label = ann["label"]
            cls_id = categories_map[label]
            preds.append([x1, y1, w, h, conf, cls_id])
        all_chunk_preds.append(((chunk_x, chunk_y), preds, original_size, image_id))

    # NMS logic
    category_boxes = defaultdict(list)
    category_confs = defaultdict(list)
    category_img_ids = defaultdict(list)
    for chunk_origin, preds, original_size, image_id in all_chunk_preds:
        chunk_x_min, chunk_y_min = chunk_origin
        original_w, original_h = original_size
        for box_x_min, box_y_min, w, h, conf, cat in preds:
            global_box_x = int(box_x_min + chunk_x_min)
            global_box_y = int(box_y_min + chunk_y_min)
            global_box_w = min(w, original_w - global_box_x)
            global_box_h = min(h, original_h - global_box_y)
            if global_box_w <= 0 or global_box_h <= 0:
                continue
            global_box = [global_box_x, global_box_y, global_box_w, global_box_h]
            category_boxes[int(cat)].append(global_box)
            category_confs[int(cat)].append(float(conf))
            category_img_ids[int(cat)].append(image_id)

    coco_predictions = {"annotations": []}
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
                    "score": round(score, 4),
                    "category_id": cat_id,
                }
            )
            ann_id += 1
            
    return coco_predictions
