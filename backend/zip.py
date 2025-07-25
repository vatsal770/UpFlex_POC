import os
import zipfile

def zip_folder(folder_path, zip_output_path):
    """
    Zips the entire folder (including subdirectories) into a zip file.

    Args:
        folder_path (str): Path to the folder you want to zip.
        zip_output_path (str): Path where the zip file will be saved.
    """
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)  # Maintain folder structure
                zipf.write(file_path, arcname)