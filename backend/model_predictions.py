import os
import json
import cv2

from pathlib import Path
import supervision as sv
from inference import get_model



def run_model_predictions_on_chunks(chunking_strategies):
    """
    For each chunking strategy and overlap, run model inference on all chunk images and save results in the same directory.
    Returns a dict with counts of chunk images per combination.
    """
    model = get_model(model_id="floorplan-o9hev/6", api_key="LLDV1nzXicfTYmlj9CMp")

    chunk_counts = {}
    for strategy, strategy_dict in chunking_strategies.items():
        for chunk_pct, overlap_dict in strategy_dict.items():
            for overlap_key, combo in overlap_dict.items():
                print(combo)
                chunk_dir = (
                    os.path.dirname(combo["predictions"][0])
                    if combo["predictions"]
                    else None
                )
                if not chunk_dir or not os.path.isdir(chunk_dir):
                    continue

                chunk_files = [
                    f
                    for f in os.listdir(chunk_dir)
                    if f.endswith(".jpg") or f.endswith(".png")
                ]
                
                allowed_classes = ["table", "chair", "table-chair"]
                chunk_counts[(strategy, chunk_pct, overlap_key)] = len(chunk_files)

                # Create corresponding output directory
                rel_path = os.path.relpath(chunk_dir, start="chunks")  # remove chunks/ prefix
                annotated_chunk_dir = os.path.join("annotated_chunks", rel_path)
                os.makedirs(annotated_chunk_dir, exist_ok=True)

                for chunk_file in chunk_files:
                    chunk_path = os.path.join(chunk_dir, chunk_file)
                    image = cv2.imread(chunk_path)
                    results = model.infer(image)[0]
                    detections = sv.Detections.from_inference(results)

                    # Save annotated image
                    bounding_box_annotator = sv.BoxAnnotator()
                    label_annotator = sv.LabelAnnotator()

                    annotated_image = bounding_box_annotator.annotate(
                        scene=image, detections=detections
                    )
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image, detections=detections
                    )

                    annotated_path = os.path.join(annotated_chunk_dir, f"annotated_{chunk_file}")
                    cv2.imwrite(annotated_path, annotated_image)

                    # Save predictions as JSON
                    annotations = []
                    for i in range(len(detections.xyxy)):
                        x_min, y_min, x_max, y_max = detections.xyxy[i]
                        class_id = detections.class_id[i]
                        confidence = float(detections.confidence[i])
                        label = (
                            detections.data["class_name"][i] if "class_name" in detections.data else str(class_id)
                        )
                        # to eliminate extra labels
                        if label not in allowed_classes:
                            continue

                        annotation = {
                            "bbox": [
                                float(x_min),
                                float(y_min),
                                float(x_max),
                                float(y_max),
                            ],
                            "label": label,
                            "confidence": confidence
                        }
                        annotations.append(annotation)

                    output_json_path = os.path.join(
                        chunk_dir, f"{Path(chunk_file).stem}.json"
                    )
                    with open(output_json_path, "w") as f:
                        json.dump(
                            {"image": chunk_path, "annotations": annotations},f,indent=4
                        )
    return chunk_counts
        
