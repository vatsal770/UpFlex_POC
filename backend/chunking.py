import os
import cv2
import json

from typing import Any, Dict
import logging

from model_predictions import run_model_predictions_on_chunks
from Custom import stitch_chunks_custom
from NMS import stitch_chunks_nms

logging.basicConfig(
    level = logging.INFO
)

logger = logging.getLogger(__name__)


def generate_results(session_dir: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    image_dir = os.path.join(session_dir, "images")
    chunk_base_dir = os.path.join(session_dir, "chunks")

    results = {}

    chunking_type = config_data.get("chunking_type", "")
    overlap_type = config_data.get("overlap_type", "")

    # Overlap and chunking config
    start_overlap = config_data.get(
        "start_overlap", config_data.get("overlap_percent", 10)
    )
    end_overlap = config_data.get("end_overlap", start_overlap)
    step_overlap = config_data.get("step_overlap", 10)
    overlap_px = config_data.get("overlap_px", 150)
    overlap_pct_dataset = config_data.get("overlap_pct", 10)
    start_pct = config_data.get("start_pct", 20)

    end_pct = config_data.get("end_pct", 40)
    step_pct = config_data.get("step_pct", 10)

    for image_file in os.listdir(image_dir):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]
        base_name = os.path.splitext(image_file)[0]
        results[image_file] = {"original": image_path, "chunking_strategies": {}}

        if chunking_type == "percentage":
            results[image_file]["chunking_strategies"]["percentage"] = {}
            for chunk_pct in range(start_pct, end_pct + 1, step_pct):
                chunk_width = max(1, (img_w * chunk_pct) // 100)
                chunk_height = max(1, (img_h * chunk_pct) // 100)
                results[image_file]["chunking_strategies"]["percentage"][
                    str(chunk_pct)
                ] = {}
                for overlap_pct in range(start_overlap, end_overlap + 1, step_overlap):
                    stride_w = max(1, int(chunk_width * (1 - overlap_pct / 100)))
                    stride_h = max(1, int(chunk_height * (1 - overlap_pct / 100)))
                    chunk_dir = os.path.join(
                        chunk_base_dir,
                        "percentage",
                        f"pct_{chunk_pct}",
                        f"overlap_{overlap_pct}",
                        base_name,
                    )
                    os.makedirs(chunk_dir, exist_ok=True)
                    predictions = []
                    chunk_id = 0
                    for y in range(0, img_h, stride_h):
                        chunk_end_y = min(img_h, y + chunk_height)
                        start_y = max(0, chunk_end_y - chunk_height)
                        for x in range(0, img_w, stride_w):
                            chunk_end_x = min(img_w, x + chunk_width)
                            start_x = max(0, chunk_end_x - chunk_width)
                            chunk = image[start_y:chunk_end_y, start_x:chunk_end_x]
                            chunk_filename = os.path.join(
                                chunk_dir, f"chunk_{chunk_id:03}.jpg"
                            )
                            cv2.imwrite(chunk_filename, chunk)
                            predictions.append(chunk_filename)
                            chunk_id += 1
                    results[image_file]["chunking_strategies"]["percentage"][
                        str(chunk_pct)
                    ][f"overlap_{overlap_pct}"] = {
                        "predictions": predictions,
                        "stitched": {"nms": "", "custom": ""},
                        "overlap_percent": overlap_pct,
                    }

        elif chunking_type == "fixed":
            chunk_width = config_data.get("chunk_width", 640)
            chunk_height = config_data.get("chunk_height", 640)
            stride_w = stride_h = 150
            chunk_dir = os.path.join(
                chunk_base_dir, f"fixed_{chunk_width}x{chunk_height}", base_name
            )
            os.makedirs(chunk_dir, exist_ok=True)
            predictions = []
            chunk_id = 0
            for y in range(0, img_h, stride_h):
                chunk_end_y = min(img_h, y + chunk_height)
                start_y = max(0, chunk_end_y - chunk_height)
                for x in range(0, img_w, stride_w):
                    chunk_end_x = min(img_w, x + chunk_width)
                    start_x = max(0, chunk_end_x - chunk_width)
                    chunk = image[start_y:chunk_end_y, start_x:chunk_end_x]
                    chunk_filename = os.path.join(chunk_dir, f"chunk_{chunk_id:03}.jpg")
                    cv2.imwrite(chunk_filename, chunk)
                    predictions.append(chunk_filename)
                    chunk_id += 1
            results[image_file]["chunking_strategies"]["fixed"] = {
                f"{chunk_width}x{chunk_height}": {
                    "predictions": predictions,
                    "stitched": {"nms": "", "custom": ""},
                }
            }

        elif chunking_type == "pixel":
            results[image_file]["chunking_strategies"]["pixel"] = {
                "chunking_skipped": True
            }

        elif chunking_type == "dataset":
            results[image_file]["chunking_strategies"]["dataset"] = {}
            for chunk_pct in range(start_pct, end_pct + 1, step_pct):
                chunk_width = max(1, (img_w * chunk_pct) // 100)
                chunk_height = max(1, (img_h * chunk_pct) // 100)
                results[image_file]["chunking_strategies"]["dataset"][
                    str(chunk_pct)
                ] = {}
                if "overlap_px" in config_data:
                    ow, oh = overlap_px, overlap_px
                    stride_w = max(1, chunk_width - ow)
                    stride_h = max(1, chunk_height - oh)
                    overlap_key = f"overlap_{ow}px"
                    overlap_val = ow
                elif "overlap_pct" in config_data:
                    ow = int(img_w * (overlap_pct_dataset / 100))
                    oh = int(img_h * (overlap_pct_dataset / 100))
                    stride_w = max(1, chunk_width - ow)
                    stride_h = max(1, chunk_height - oh)
                    overlap_key = f"overlap_{overlap_pct_dataset}pct"
                    overlap_val = overlap_pct_dataset
                else:
                    continue
                chunk_dir = os.path.join(
                    chunk_base_dir,
                    "dataset",
                    f"pct_{chunk_pct}",
                    overlap_key,
                    base_name,
                )
                os.makedirs(chunk_dir, exist_ok=True)
                predictions = []
                chunk_id = 0
                for y in range(0, img_h, stride_h):
                    chunk_end_y = min(img_h, y + chunk_height)
                    start_y = max(0, chunk_end_y - chunk_height)
                    for x in range(0, img_w, stride_w):
                        chunk_end_x = min(img_w, x + chunk_width)
                        start_x = max(0, chunk_end_x - chunk_width)
                        chunk = image[start_y:chunk_end_y, start_x:chunk_end_x]
                        chunk_filename = os.path.join(
                            chunk_dir, f"chunk_{chunk_id:03}.jpg"
                        )
                        cv2.imwrite(chunk_filename, chunk)
                        predictions.append(chunk_filename)
                        chunk_id += 1
                results[image_file]["chunking_strategies"]["dataset"][str(chunk_pct)][
                    overlap_key
                ] = {
                    "predictions": predictions,
                    "stitched": {"nms": "", "custom": ""},
                    (
                        "overlap_px"
                        if "overlap_px" in config_data
                        else "overlap_percent"
                    ): overlap_val,
                }

    # Before running stitching:
    original_sizes = {}
    image_name_to_id = {}
    image_id_counter = 1

    # 2. Run model predictions for all chunking strategies
    for image_file, image_info in results.items():
        chunking_strategies = image_info["chunking_strategies"]
        chunk_counts = run_model_predictions_on_chunks(chunking_strategies)
        image_info["chunk_counts"] = chunk_counts

        # 3. For each chunking strategy/combination, run stitching and update paths
        for strategy, strategy_dict in chunking_strategies.items():
            for chunk_pct, overlap_dict in strategy_dict.items():
                for overlap_key, combo in overlap_dict.items():
                    chunk_dir = (
                        os.path.dirname(combo["predictions"][0])
                        if combo["predictions"]
                        else None
                    )
                    print(chunk_dir)
                    if not chunk_dir or not os.path.isdir(chunk_dir):
                        continue

                    for chunk_path in combo["predictions"]:
                        filename = os.path.basename(chunk_path)

                        # if "_x" not in filename or "_y" not in filename:
                        #     continue

                        original_image_name = filename

                        if original_image_name not in original_sizes:
                            # Load actual original image from disk to get dimensions
                            try:
                                image_path = os.path.join(chunk_dir, original_image_name)
                                img = cv2.imread(image_path)
                                h, w = img.shape[:2]
                                original_sizes[original_image_name] = (w, h)

                                image_name_to_id[original_image_name] = image_id_counter
                                image_id_counter += 1
                            except Exception as e:
                                print(f"⚠️ Failed to load image {original_image_name}: {e}")
                    

                    categories_map = [
                        {"name": "chair", "id": 100},
                        {"name": "table", "id": 101}, 
                        {"name": "table-chair", "id": 103}, 
                    ]


                    # NMS stitching
                    stitched_nms_path = os.path.join(chunk_dir, "stitched_nms.json")
                    # Placeholder: you must provide original_sizes, image_name_to_id, categories_map
                    stitched_nms = stitch_chunks_nms(chunk_dir, original_sizes, image_name_to_id, categories_map)
                    with open(stitched_nms_path, "w") as f:
                        json.dump(stitched_nms, f, indent=2)
                    combo["stitched"]["nms"] = stitched_nms_path
                    # Custom stitching placeholder
                    stitched_custom_path = os.path.join(
                        chunk_dir, "stitched_custom.json"
                    )
                    stitched_custom = stitch_chunks_custom(chunk_dir, original_sizes, image_name_to_id, categories_map)
                    with open(stitched_custom_path, "w") as f:
                        json.dump(stitched_custom, f, indent=2)
                    combo["stitched"]["custom"] = stitched_custom_path


    # make the logs of the results
    logger.info(results)
    return results

