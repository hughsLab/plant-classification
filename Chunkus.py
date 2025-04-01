import os
import shutil
from math import ceil

# Paths for the original dataset and the target chunks
source_dir = r"D:\PLANT\Train"
destination_dir = r"D:\PLANT\Chunkus"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all folders in the source directory
folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]
total_folders = len(folders)
print(f"Total number of folders: {total_folders}")

# Calculate how many folders per chunk (dividing as evenly as possible)
folders_per_chunk = ceil(total_folders / 10)

# Distribute the folders across 10 chunks
for chunk_number in range(1, 11):
    # Create a new chunk folder
    chunk_folder_path = os.path.join(destination_dir, f"Chunk_{chunk_number}")
    if not os.path.exists(chunk_folder_path):
        os.makedirs(chunk_folder_path)

    # Determine which folders to move to this chunk
    start_index = (chunk_number - 1) * folders_per_chunk
    end_index = min(start_index + folders_per_chunk, total_folders)
    folders_to_move = folders[start_index:end_index]

    # Copy each folder to the chunk folder
    for folder in folders_to_move:
        src_path = os.path.join(source_dir, folder)
        dest_path = os.path.join(chunk_folder_path, folder)
        shutil.copytree(src_path, dest_path)

print("Folders have been divided into 10 chunks successfully!")
