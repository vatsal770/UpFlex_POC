import os
import json
import re
import logging 
import cv2
import shutil

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def boxes_intersect_enough(box1, box2, min_overlap=15, comparison_thresh=5, containment_thresh=2):
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

    if abs(y2_max - y1_max) <= comparison_thresh and abs(y2_min - y1_min) <= comparison_thresh:
        y_overlap = 0
    elif abs(x2_max - x1_max) <= comparison_thresh and abs(x2_min - x1_min) <= comparison_thresh:
        x_overlap = 0

    # Containment logic: is box1 inside box2 or vice versa?
    def is_contained(inner, outer, tol):
        ix_min, iy_min, iw, ih = inner
        ix_max, iy_max = ix_min + iw, iy_min + ih

        ox_min, oy_min, ow, oh = outer
        ox_max, oy_max = ox_min + ow, oy_min + oh

        return (
            ix_min >= ox_min - tol and
            iy_min >= oy_min - tol and
            ix_max <= ox_max + tol and
            iy_max <= oy_max + tol
        )

    containment = is_contained(box1, box2, containment_thresh) or is_contained(box2, box1, containment_thresh)

    # Return True if enough overlap OR one box is inside the other
    return (x_overlap >= min_overlap or y_overlap >= min_overlap) or containment


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

        # Only write label once per category
        if label not in drawn_labels:
            cv2.putText(img, label, (10, 30 + 25 * len(drawn_labels)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            drawn_labels.add(label)

    out_path = os.path.join(output_dir, "annotated_Final_Custom.jpg")
    cv2.imwrite(out_path, img)
    logger.info(f"✅ Saved annotated image to: {out_path}")




def stitch_chunks_custom(
    session_dir: str,
    predictions_dir: List[str],
    image_path: str,
    json_formats: List[str],
    merge_thresh = 15,
    comparison_thresh = 5,
    containment_thresh = 2,
):
    """
    Custom stitching logic using proximity-based union of boxes.
    """

    category_map = {
        "chair": 100,
        "table": 101,
        "table-chair": 103    
        }
    
    inverse_category_map = {
        100: "chair",
        101: "table",
        103: "table-chair"
    }
    
    parts = os.path.normpath(predictions_dir).split(os.sep)
    chunk_pct, overlap = parts[-4], parts[-3]

    image_name = (Path(image_path)).stem

    image = cv2.imread(image_path)
    original_h, original_w  = image.shape[:2]
    updated_image_name = Path(image_path).stem + f"_{chunk_pct}_{overlap}.jpg"
    


############ Defining the base JSON formats ################################

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
                "file_name": updated_image_name,
                "width": original_w, 
                "height":original_h,
                "date_captured": "2024-11-25T04:08:53+00:00"
            }
        ]
    }

    # In createML JSON format
    createML_predictions = [
        {
            "image": updated_image_name
        }
    ]
###############################################################



    logger.info("Prediction directory: %s", predictions_dir)

    # Load chunk predictions
    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(predictions_dir, file)

        with open(file_path, "r") as f:
            chunk_data = json.load(f)

        # extracting the filename of the chunks
        chunk_image = chunk_data["images"][0]
        filename = chunk_image["file_name"]    # Extract filename

        annotations = chunk_data["annotations"]

        # extracting the start coordinates of the chunk image from the filename
        match = re.search(r"_x(\d+)_y(\d+)\.jpg", filename)
        if not match:
            logger.info("Match not Found while tracing back the Chunk image!!!!!!")
            continue

        chunk_x = int(match.group(1))
        chunk_y = int(match.group(2))

        # logger.info(f"{chunk_x} and {chunk_y}")

        # merging the annotation predictions of all chunk images
        preds = []
        for ann in annotations:
            x1, y1, w, h = ann["bbox"]
            cls_id = ann["category_id"]
            preds.append([x1, y1, w, h, cls_id])

        all_chunk_preds.append(((chunk_x, chunk_y), preds, 1))

    # Checking total predictions count
    preds_count = 0
    unique_categories = set()
    for chunk in all_chunk_preds:
        preds = chunk[1]  # ensure we get preds
        for p in preds:
            preds_count += 1
            unique_categories.add(p[4])

    logger.info(f"🔍 Total predictions count: {preds_count}")


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

    logger.info(f"Total Boxes negelected gue to Gloabal merging: {total_boxes_neglected}")
    logger.info(f"cat_img_boxes keys: {list(cat_img_boxes.keys())}")
    for cat_id, image_boxes_dict in cat_img_boxes.items():
        logger.info(f"Category {cat_id}: {[len(v) for v in image_boxes_dict.values()]}")


    # Logger Debugging part which calculates total number of chairs, tables and other items in the image
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



    # Create Annotations list in all json formats 
    coco_predictions["annotations"] = []
    createML_predictions[0]["annotations"] = []
    
    ann_id = 1

    # Group and merge
    for cat_id, image_boxes_dict in cat_img_boxes.items():
        for image_id, boxes in image_boxes_dict.items():
            n = len(boxes)
            uf = UnionFind(n)
            for i in range(n):
                for j in range(i + 1, n):
                    # check if the two boxes intersect enough
                    if boxes_intersect_enough(boxes[i], boxes[j], min_overlap=merge_thresh, comparison_thresh=comparison_thresh, containment_thresh=containment_thresh):
                        uf.union(i, j)

            group_dict = defaultdict(list)
            for idx in range(n):
                root = uf.find(idx)
                group_dict[root].append(boxes[idx])

            for group_boxes in group_dict.values():
                merged_box = merge_boxes(group_boxes)

                # adding coco-annotations
                coco_predictions["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x, 2) for x in merged_box],
                    "area": int(merged_box[2]*merged_box[3]),
                    "segmentation": [],
                    "iscrowd": 0
                })
                ann_id += 1

                if "createML" in json_formats:
                    # Extract bounding box values
                    x_min = round(merged_box[0], 2)
                    y_min = round(merged_box[1], 2)
                    width = round(merged_box[2], 2)
                    height = round(merged_box[3], 2)

                    # Convert to center-based coordinates
                    center_x = round(x_min + (width / 2), 2)
                    center_y = round(y_min + (height / 2), 2)

                    # adding createML-annotations
                    createML_predictions[0]["annotations"].append({
                        "label": inverse_category_map[cat_id],
                        "coordinates": {
                            "x": center_x,
                            "y": center_y,
                            "width": width,
                            "height": height
                        }
                    })



    # Dumping coco-annotations
    custom_json_dir_coco = os.path.join(session_dir, "COCO", image_name, "Dataset", chunk_pct, overlap, "full_image", "annotated_json_full")
    os.makedirs(custom_json_dir_coco, exist_ok=True)
    # logger.info(f"Custom Directory {custom_dir}")
    output_json_path_coco = os.path.join(custom_json_dir_coco, f"stitched_{chunk_pct}_{overlap}.json")
    with open(output_json_path_coco, "w") as f:
        json.dump(coco_predictions, f, indent=4)

    if "createML" in json_formats:
        # Dumping createML-annotations
        custom_json_dir_createML = os.path.join(session_dir, "createML", image_name, "Dataset", chunk_pct, overlap, "full_image", "annotated_json_full")
        os.makedirs(custom_json_dir_createML, exist_ok=True)
        # logger.info(f"Custom Directory {custom_dir}")
        output_json_path_createML = os.path.join(custom_json_dir_createML, f"stitched_{chunk_pct}_{overlap}.json")
        with open(output_json_path_createML, "w") as f:
            json.dump(createML_predictions, f, indent=4)



    # Copying real full image
    for json_format in json_formats:
        copy_real_img_dir = os.path.join(session_dir, json_format, image_name, "Dataset", chunk_pct, overlap, "full_image", "images")
        os.makedirs(copy_real_img_dir, exist_ok=True)
        # Copy file
        real_img_dir = os.path.join(copy_real_img_dir, updated_image_name)
        shutil.copy(image_path, real_img_dir)


    # defining custom_dir to save the annotated_full_image
    custom_img_dir = os.path.join(session_dir, image_name, "Visualize", chunk_pct, overlap, "annotated_full_images")
    draw_predictions_single_image(coco_predictions, image_path, custom_img_dir)

    # Copying annotated full image
    for json_format in json_formats:
        annotated_image_path = os.path.join(custom_img_dir, "annotated_Final_Custom.jpg")
        copy_annotated_full_image_dir = os.path.join(session_dir, json_format, image_name, "Visualize", chunk_pct, overlap, "annotated_full_image")
        os.makedirs(copy_annotated_full_image_dir, exist_ok=True)
        # Copy file
        annotated_img_dir_coco = os.path.join(copy_annotated_full_image_dir, updated_image_name)
        shutil.copy(annotated_image_path, copy_annotated_full_image_dir)
    
