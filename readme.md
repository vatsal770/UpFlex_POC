# UpFlex Floor-planning

This project implements a full backend pipeline for detecting chairs, tables, and clusters from large floor plan images using a chunk-based detection and stitching strategy. The pipeline supports custom chunking, model inference, postprocessing, and result visualization.

---

## Features

* **Chunking** of large floor plan images with overlap strategies (percentage or pixels).
* **Inference** using our pre-trained rfdetr model via the `inference` API.
* **Coordinate Merging** to convert chunk predictions into global image coordinates.
* **Stitching** using proximity-based Union-Find algorithm to merge overlapping boxes.
* **Support for multiple JSON export formats**: `COCO`, `CreateML`, and others.
* **Zipped Output** containing images, HTML visualizations, and annotations.

---

## 📁 Project Structure

```
.
├── backend
│   └── uploads                   # Folder where user files and results are saved
│   └── chunking.py               # Chunking logic
│   └── model_predictions.py      # Run model inference on chunks
│   └── custom.py                 # Stitching logic and utilities
│   └── zip.py                    # Handles output zipping
│   └── pipeline.py               # Orchestrates full processing pipeline
│   └── main.py                   # FastAPI entry point
│
├── frontend
    └── pages                     
    │   └── chunk_preview.py      # Streamlit UI page for previewing chunks
    │   └── full_analysis.py      # Full result visualizer page (Streamlit)
    └── app.py                    # app which assembles the pages

```

---

## ⚙️ How It Works

### 1. Chunking Logic

* Images are split into smaller sized chunks.
* Supports two types:

  * `percentage` based (start-end percent)
  * `pixel` based (height-width in pixels)

### 2. Overlap Logic

* Image objects may be chunked from-between. To encounter that, certain overlap is introduced with chunking.
* Supports three types:

  * `percentage` based (start-end percent)
  * `dataset_pct` based (with a standard percentage obtained through data analysis)
  * `dataset_px` based (with a standard pixel value obtained through data analysis)

### 3. 🔍 Model Predictions

* Each chunk is sent to the selected model using `get_model()` interface.
* Only allowed(selected) classes are passed into the annotations JSON file, to eliminate unrequired classes.
* Predictions are collected and saved in COCO-style JSON per chunk.

### 4. Stitching (Postprocessing)

Once model predictions are generated for each chunked image, the system performs a stitching phase to unify all predictions back into the coordinate space of the original full-resolution image. This ensures that overlapping predictions across chunk boundaries are resolved properly.

#### 1. Starting Workflow
* For each chunk, prediction results are loaded from corresponding .json files (usually in COCO format). The associated metadata (e.g., x, y of the chunk within the original image) is used to translate the box to global coordinates.
* If a box in the chunk is [x1, y1, w, h] and the chunk was located at (chunk_x, chunk_y) in the original image.

    ```python
    global_x = x1 + chunk_x
    global_y = y1 + chunk_y
    ```

#### 2. Grouping
* Before merging, predictions are grouped by:

    ```
    category_id → Each object category is processed independently.
    image_id (usually always 0 in single image inference, but supports multiple images).
    ```

#### 3. Union-Find Merging Logic

* The system uses a Union-Find (Disjoint Set Union) structure to cluster overlapping boxes:

    *  For each pair of boxes in the same group, a comparison is done using:

        ```python
        boxes_intersect_enough(boxA, boxB, ...)
        ```
    * Two boxes are merged (grouped together) if they:

        * Overlap horizontally or vertically beyond a certain threshold (min_overlap).
        ```python
        x_min = min([b[0] for b in group])
        y_min = min([b[1] for b in group])
        x_max = max([b[0] + b[2] for b in group])
        y_max = max([b[1] + b[3] for b in group])
        # Calculate overlapping area
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        ```
        * Comparison_thresh is applied to avoid false mergings.
        ```python
        # Check for false merging cases (eg- x_overlap<merge_threshold , y_overlap)
        if abs(y2_max - y1_max) <= comparison_thresh and abs(y2_min - y1_min) <= comparison_thresh:
            y_overlap = 0
        elif abs(x2_max - x1_max) <= comparison_thresh and abs(x2_min - x1_min) <= comparison_thresh:
            x_overlap = 0
        ```
        * One box is fully contained within another within a containment_thresh.
        ```python
        boxA.x_min ≥ boxB.x_min - tol
        boxA.y_min ≥ boxB.y_min - tol
        boxA.x_max ≤ boxB.x_max + tol
        boxA.y_max ≤ boxB.y_max + tol

        return (x_overlap >= min_overlap or y_overlap >= min_overlap) or containment
        ```

    * The Union-Find structure merges indices that satisfy the above criteria.
        ```python
        Final merged box = [x_min, y_min, x_max - x_min, y_max - y_min]
        ```
    * At the end, each connected component (group) of boxes is merged.

### 5. 📦 Zipping

* (Final+Chunked) annotations, and (Final+Chunked) images are bundled into per-format ZIPs.

---

## 🧪 FastAPI Endpoints

### `/only_chunking`

* Accepts images + config
* Performs chunking and export only
* Returns `session_dir` path

### `/upload_and_process`

* Accepts images + config
* Runs chunking, prediction, stitching, export
* Returns `session_dir` path

---

## ⚙️ Configuration Parameters

The config file for full-analysis should include:

```json
{
  "chunking_type": "percentage",         // or "pixel",
  "chunking_params": { ... },             // percent or pixel ranges
  "overlap_type": "percentage",          // or "dataset_pct" or "dataset_px,
  "overlap_params": { ... },
  "model_id": "rfdetr-base",             // model name
  "api_key": "...",
  "selected_classes": ["chair", "table", "table-chair"],
  "json_formats": ["COCO", "CreateML"],
  "stitching_type": "custom",           // currently only custom is supported
  "stitching_params": {
    "merge_thresh": 15,
    "comparison_thresh": 5,
    "containment_thresh": 2
  }
}
```

---

## 🧪 Workflow

1. **User uploads** a floor plan images via Streamlit UI.
2. System generates chunks, and stores metadata for each.
3. Detection Model is run on each chunk and predictions(annotations) are saved in JSON format.
4. Predictions are merged and stitched globally.
5. Results are visualized and zipped.

---

## ✅ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Output Structure

For each image (Chunk Analysis):

```
uploads/
└── ChunkPreview/
    └── Generated_chunks/
        ├── [image_name]/
        │   ├── chunk_pct/overlap/
        │       └── chunk data (annotations_JSON, images, metadata)
        ├── [image_name]/
        │   ├── ...
```

For each image and format (Full Analysis):

```
uploads/
└── Visualization/
    └── COCO/
        └── [image_name]/
            ├── Dataset/chunk_pct/overlap/
            │   └── chunk data (annotations_JSON, images, metadata)
            │   └── full image data (annotations_JSON, image)
            ├── Visualize/chunk_pct/overlap/
            │   └── annotated chunk data 
            │   └── annotated full image 
    └── CreateML/
        └── ...
```

---

## 📌 Notes

* The metadata per chunk includes its starting position coordinates(`x`, `y`), chunk\_id, chunk\_type, chunk\_height/width, overlap\_type and overlap\_size.
* Annotated chunk images are also saved for debugging/visualization.
* Chunk IDs like `chunk_1`, `chunk_2` are used to correlate metadata and image chunk files.

---
