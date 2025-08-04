import json
import logging
import os
from typing import List
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from chunking import ChunkProcessor
from pipeline import ProcessingPipeline


# Configure logging once for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Create FastAPI app instance
app = FastAPI()

# Endpoint to handle only chunking
@app.post("/only_chunking")
async def only_chunking(
    files: List[UploadFile] = File(...), config: str = Form(...)
):
    try:
        session_name = "ChunkPreview"
        uploads_dir = os.path.join("./uploads", session_name)

        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)
        os.makedirs(uploads_dir)

        image_dir = os.path.join(uploads_dir, "user_images")
        os.makedirs(image_dir, exist_ok=True)

        for file in files:
            file_name = file.filename
            with open(os.path.join(image_dir, file_name), "wb") as f:
                f.write(await file.read())

        # Save configurations sent by the user
        try:
            config_data = json.loads(config)
        except json.JSONDecodeError as e:
            logger.error("Invalid config JSON: %s", e)
            return JSONResponse(status_code=400, content="Invalid configuration JSON")
        
        config_path = os.path.join(uploads_dir, "config.json")  # defining config file path
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Extract chunking configurations
        chunking_type = config_data.get("chunking", {}).get("type", "")
        chunk_params = config_data.get("chunking", {}).get("params", "")

        # Overlap configurations
        overlap_type = config_data.get("overlap", {}).get("type", "")
        overlap_params = config_data.get("overlap", {}).get("params", {})

        # Call only chunking logic
        chunk_processor = ChunkProcessor(uploads_dir, chunking_type, chunk_params, overlap_type, overlap_params)
        chunk_processor.generate_chunks()
        session_dir = os.path.abspath(uploads_dir)
        logger.info(f"Session directory: {session_dir}")

        return JSONResponse(status_code=200, content={"session_path": session_dir})
    
    except Exception as e:
        logger.error("Error processing chunk preview: %s", str(e))
        return JSONResponse(status_code=500, content="Internal Server Error")


# Endpoint to handle full implementation
@app.post("/upload_and_process")
async def upload_and_process(
    files: List[UploadFile] = File(...), config: str = Form(...)
):
    try:
        # Session Name to have a common directory for each upload.
        session_name = "Visualization"
        uploads_dir = os.path.join("./uploads", session_name)

        if os.path.exists(uploads_dir):
            shutil.rmtree(uploads_dir)

        os.makedirs(uploads_dir)

        # Save images sent by the user.
        image_dir = os.path.join(uploads_dir, "user_images")
        os.makedirs(image_dir, exist_ok=True)
        for file in files:
            if file.filename:
                file_name: str = file.filename
            else:
                raise ValueError("Invlaid file name.")

            img_path: str = os.path.join(image_dir, file_name)
            with open(img_path, "wb") as f:
                f.write(await file.read())

        # Save configurations sent by the user.
        try:
            config_data = json.loads(config)
        except json.JSONDecodeError as e:
            logger.error("Invalid config JSON: %s", e)
            return JSONResponse(status_code=400, content="Invalid configuration JSON")

        config_path = os.path.join(uploads_dir, "config.json")  # defining config file pathz
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # extracting model_id, api_key, json_formats, selected_classes from the config file
        model_id = config_data["model_id"]["params"]["model_selected"]
        api_key = config_data["api_key"]["params"]["api_selected"]
        json_formats = config_data["json_formats"]["params"]["formats_selected"]
        selected_classes = config_data["allowed_classes"]["params"]["selected"]

        # Extract chunking configurations
        chunking_type = config_data.get("chunking", {}).get("type", "")
        chunk_params = config_data.get("chunking", {}).get("params", "")

        # Overlap configurations
        overlap_type = config_data.get("overlap", {}).get("type", "")
        overlap_params = config_data.get("overlap", {}).get("params", {})

        # Stitching configurations
        stitching_type = config_data.get("stitching", {}).get("type", "")
        stitching_params = config_data.get("stitching", {}).get("params", {})

        # Generate results from the created pipeline
        pipeline = ProcessingPipeline(uploads_dir, model_id, api_key, selected_classes, json_formats, chunking_type, chunk_params, overlap_type, overlap_params, stitching_type, stitching_params)
        pipeline.run_chunking_for_all_images()
        pipeline.run_predictions_for_all_images()
        pipeline.run_stitch_for_all_images()
        pipeline.run_zip_all_formats()
        
        session_dir = os.path.abspath(uploads_dir)
        logger.info(f"Session directory: {session_dir}")
        
        return JSONResponse(status_code=200, content={"session_path": session_dir})


    except Exception as e:
        logger.error("Error processing files: %s", str(e))
        return JSONResponse(status_code=500, content="Internal Server Error")
