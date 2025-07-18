import os
import json
import re
import logging 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def boxes_intersect_enough(box1, box2, min_overlap=15):
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    if x_overlap > 0 and y_overlap == 0:
        x_overlap = 0
    if y_overlap > 0 and x_overlap == 0:
        y_overlap = 0

    if abs(y2_max - y1_max) <= 10 and abs(y2_min - y1_min) <= 10:
        y_overlap = 0
    elif abs(x2_max - x1_max) <= 10 and abs(x2_min - x1_min) <= 10:
        x_overlap = 0

    return x_overlap >= min_overlap or y_overlap >= min_overlap


def merge_boxes(boxes):
    x_min = min(b[0] for b in boxes)
    y_min = min(b[1] for b in boxes)
    x_max = max(b[0] + b[2] for b in boxes)
    y_max = max(b[1] + b[3] for b in boxes)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv



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

        out_path = os.path.join(output_dir, f"annotated_Final_Custom")
        cv2.imwrite(out_path, img)
        logger.info(f"Image_saved_successfully!!!!!!!!!!!")




def stitch_chunks_custom(
    predictions_dir,
    image_path,
    merge_thresh = 15
):
    """
    Custom stitching logic using proximity-based union of boxes.
    """
    image = cv2.imread(image_path)
    original_h, original_w  = image.shape[:2]
    image_name = os.path.basename(image_path)


    coco_predictions = {
        "images": [
            {
                "id": 1, 
                "file_name": image_name,
                "width": original_w, 
                "height":original_h
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

    logger.info("Prediction directory: %s", predictions_dir)

    # Load chunk predictions
    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(predictions_dir, file)
        # logger.info(file_path)

        with open(file_path, "r") as f:
            chunk_data = json.load(f)

        chunk_image = chunk_data["image"]

        filename = Path(chunk_image.split("_x")[0] + ".jpg").stem

        annotations = chunk_data["annotations"]

        match = re.search(r"_x(\d+)_y(\d+)\.jpg", chunk_image)
        if not match:
            logger.info("Match not Found while tracing back the Chunk image!!!!!!")
            continue

        chunk_x = int(match.group(1))
        chunk_y = int(match.group(2))

        logger.info(f"{chunk_x} and {chunk_y}")

        # # Now verify this matches
        # if filename not in image_name_to_id:
        #     logger.warning(f"❌ Skipping chunk: filename '{filename}' not found in mapping")
        #     continue

        preds = []
        for ann in annotations:
            x1, y1, w, h = ann["bbox"]
            cls_id = ann['category_id']
            preds.append([x1, y1, w, h, cls_id])

        all_chunk_preds.append(((chunk_x, chunk_y), preds, 1))

    # ✅ Checking total predictions count
    preds_count = 0
    unique_categories = set()
    for chunk in all_chunk_preds:
        preds = chunk[1]  # ensure we get preds
        for p in preds:
            preds_count += 1
            unique_categories.add(p[4])

    logger.info(f"🔍 Total predictions count: {preds_count}")
    # logger.info(f"📊 All categories present: {sorted(unique_categories)}")

    # Merge logic using Union-Find
    cat_img_boxes = defaultdict(lambda: defaultdict(list))

    total_boxes = 0
    chairs, tables, clusters = 0, 0, 0

    total_boxes_neglected = 0
    for chunk_origin, preds, image_id in all_chunk_preds:
        chunk_x, chunk_y = chunk_origin

        for box_x, box_y, w, h, cat_id in preds:
            global_x = int(box_x + chunk_x)
            global_y = int(box_y + chunk_y)
            global_w = min(w, original_w - global_x)
            global_h = min(h, original_h - global_y)

            if global_w <= 0 or global_h <= 0:
                total_boxes_neglected += 1
                logger.info(f"Problematic Coordinates: {box_x}, {chunk_x},{box_y}, {chunk_y}")
                continue

            box = [global_x, global_y, global_w, global_h]
            cat_img_boxes[cat_id][image_id].append(box)

    logger.info(f"Toatal Boxes negelected gue to Gloabal merging: {total_boxes_neglected}")
    logger.info(f"cat_img_boxes keys: {list(cat_img_boxes.keys())}")
    for cat_id, image_boxes_dict in cat_img_boxes.items():
        logger.info(f"Category {cat_id}: {[len(v) for v in image_boxes_dict.values()]}")


    # Logger Debugging part

    for cat_id, image_boxes_dict in cat_img_boxes.items():  # category_id → image_id → boxes
        for boxes in image_boxes_dict.values():
            count = len(boxes)
            total_boxes += count

            if cat_id == 100:     # chair
                chairs += count
            elif cat_id == 101:   # table
                tables += count
            elif cat_id == 103:   # table-chair or cluster
                clusters += count

    logger.info(f"📦 Total predicted boxes across all chunks: {total_boxes}")
    logger.info(f"🪑 Chairs: {chairs} | 🧱 Tables: {tables} | 🧩 Clusters: {clusters}")



    # Group and merge
    coco_predictions["annotations"] = []
    ann_id = 1
    for cat_id, image_boxes_dict in cat_img_boxes.items():
        for image_id, boxes in image_boxes_dict.items():
            n = len(boxes)
            uf = UnionFind(n)
            for i in range(n):
                for j in range(i + 1, n):
                    if boxes_intersect_enough(boxes[i], boxes[j], min_overlap=merge_thresh):
                        uf.union(i, j)

            group_dict = defaultdict(list)
            for idx in range(n):
                root = uf.find(idx)
                group_dict[root].append(boxes[idx])

            for group_boxes in group_dict.values():
                merged_box = merge_boxes(group_boxes)
                coco_predictions.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [round(x, 2) for x in merged_box],
                    "category_id": cat_id
                })
                ann_id += 1

    output_json_path = os.path.join(predictions_dir, "stitched_predictions_custom.json")
    with open(output_json_path, "w") as f:
        json.dump(coco_predictions, f, indent=4)

    draw_predictions_single_image(coco_predictions, image_path, predictions_dir)

    
