import os
import tarfile
from tqdm import tqdm

# File paths
source_file = r"C:\Users\USER\Desktop\PLANT\PlantCLEF2017Train2Web.tar.gz"
destination_folder = r"C:\Users\USER\Desktop\PLANT"

# Function to extract with progress bar
def extract_tar_gz(file_path, dest_folder):
    with tarfile.open(file_path, 'r:gz') as tar:
        # Get total number of members to track progress
        members = tar.getmembers()
        total_files = len(members)
        
        # Progress bar setup
        with tqdm(total=total_files, desc=f"Extracting {os.path.basename(file_path)}", unit="file") as progress_bar:
            for member in members:
                tar.extract(member, path=dest_folder)
                progress_bar.update(1)

# Ensure the file exists before extraction
if os.path.exists(source_file):
    extract_tar_gz(source_file, destination_folder)
else:
    print(f"File not found: {source_file}")
