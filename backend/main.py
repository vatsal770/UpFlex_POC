import json
import logging
import os
from typing import List
import shutil

from chunking import generate_results
from only_chunking import generate_chunks
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        config_data = json.loads(config)
        config_path = os.path.join(uploads_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Call only chunking logic
        session_dir = generate_chunks(uploads_dir, config_data)
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
        config_data = json.loads(config)
        config_path = os.path.join(uploads_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Generate results
        session_dir = generate_results(uploads_dir, config_data)
        session_dir = os.path.abspath(uploads_dir)
        logger.info(f"Session directory: {session_dir}")
        
        return JSONResponse(status_code=200, content={"session_path":session_dir})


    except Exception as e:
        logger.error("Error processinf files: %s", str(e))
        return JSONResponse(status_code=500, content="Internal Server Error")
