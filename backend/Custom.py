import os
import json
import re
from collections import defaultdict


def stitch_chunks_custom(
    predictions_dir,
    original_sizes,
    image_name_to_id,
    categories_map,
    merge_thresh=15
):
    """
    Custom stitching logic using proximity-based union of boxes.
    """

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


    # Load chunk predictions
    all_chunk_preds = []
    for file in os.listdir(predictions_dir):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(predictions_dir, file)
        with open(file_path, "r") as f:
            chunk_data = json.load(f)

        chunk_image = chunk_data["image"]
        if isinstance(chunk_image, list):  # handle if wrapped in list
            chunk_image = chunk_image[0]

        annotations = chunk_data["annotations"]
        filename = os.path.basename(chunk_image["file_name"])

        match = re.search(r"_x(\d+)_y(\d+)\.jpg", filename)
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
            label = ann["label"]
            cls_id = categories_map[label]
            preds.append([x1, y1, w, h, cls_id])

        all_chunk_preds.append(((chunk_x, chunk_y), preds, original_size, image_id))

    # Merge logic using Union-Find
    cat_img_boxes = defaultdict(lambda: defaultdict(list))
    for chunk_origin, preds, original_size, image_id in all_chunk_preds:
        chunk_x, chunk_y = chunk_origin
        original_w, original_h = original_size

        for box_x, box_y, w, h, cat_id in preds:
            global_x = int(box_x + chunk_x)
            global_y = int(box_y + chunk_y)
            global_w = min(w, original_w - global_x)
            global_h = min(h, original_h - global_y)

            if global_w <= 0 or global_h <= 0:
                continue

            box = [global_x, global_y, global_w, global_h]
            cat_img_boxes[cat_id][image_id].append(box)

    # Group and merge
    coco_predictions = {"annotations": []}
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
                coco_predictions["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "bbox": [round(x, 2) for x in merged_box],
                    "category_id": cat_id
                })
                ann_id += 1

    return coco_predictions
