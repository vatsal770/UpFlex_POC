import logging
import os

from typing import Any, Dict
from zip import zip_folder      # import zip_folder from zip.py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_zip_all_formats(
    session_dir: str, 
    config_data: Dict[str, Any]
):
    """
    Run zipping for all formats in the session directory.

    Args:
        session_dir (str): Path to the session directory.
        config_data (Dict[str, Any]): Configuration data containing model and chunking parameters.

    Returns:
        None
    """

    logger.info("Starting zipping for all formats in session directory: %s", session_dir)

    # extracting model_id, api_key, json_formats, selected_classes from the config file
    json_formats = config_data["json_formats"]["params"]["formats_selected"]

    # create zipped file per json format type
    for json_format in json_formats:
        folder_path = os.path.join(session_dir, json_format)     # input_folder
        zip_output_path = folder_path + ".zip"    # output_zip_path
        zip_folder(folder_path, zip_output_path)
        logger.info(f"✅ Folder '{folder_path}' zipped successfully at: {zip_output_path}")