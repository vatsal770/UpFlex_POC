import json
import os
import uuid
from chunking import generate_results
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/upload_and_process")
async def upload_and_process(
    files: List[UploadFile] = File(...), config: str = Form(...)
):
    session_id = str(uuid.uuid4())
    save_dir = os.path.join("./uploads", session_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save images
    image_dir = os.path.join(save_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    for file in files:
        img_path = os.path.join(image_dir, file.filename)
        with open(img_path, "wb") as f:
            f.write(await file.read())

    # Save config
    config_data = json.loads(config)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    # Generate results
    results = generate_results(save_dir, config_data)
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return JSONResponse({"results_path": results_path})
