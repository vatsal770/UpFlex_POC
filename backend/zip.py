import os
import zipfile

class ZipMaker:
    def __init__(self, session_dir: str):
        self.session_dir = session_dir

    def zip_folder(self, input_folder_name: str, zip_output_name: str):
        """
        Zips the entire folder (including subdirectories) into a zip file.

        Args:
            input_folder_name (str): Name to the folder you want to zip.
            zip_output_name (str): Name of the zip file to be saved.
        """

        # define full paths to the input_folder and output zip_folder
        input_folder_path = os.path.join(self.session_dir, input_folder_name)
        zip_output_path = os.path.join(self.session_dir, zip_output_name)
        
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(input_folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, input_folder_path)  # Maintain folder structure
                    zipf.write(file_path, arcname)