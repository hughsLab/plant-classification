import os
from PIL import Image, UnidentifiedImageError

# Define the directories to search
directories = [
    r"C:\Users\USER\Desktop\PLANT\test",
    r"C:\Users\USER\Desktop\PLANT\Train",
    r"C:\Users\USER\Desktop\PLANT\Validation"
]

# Function to find files with extensions other than .jpg
def find_non_jpg_files(directory):
    non_jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith('.jpg'):
                non_jpg_files.append(os.path.join(root, file))
    return non_jpg_files

# Function to check if a JPG image is valid (not corrupted)
def validate_jpg_files(directory):
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Check if the file is a valid image
                except (UnidentifiedImageError, IOError):
                    corrupted_files.append(file_path)
    return corrupted_files

# Iterate over each directory and collect non-JPG files
all_non_jpg_files = []
corrupted_jpg_files = []
for directory in directories:
    # Find non-JPG files
    non_jpg_files = find_non_jpg_files(directory)
    all_non_jpg_files.extend(non_jpg_files)

    # Validate JPG files
    corrupted_files = validate_jpg_files(directory)
    corrupted_jpg_files.extend(corrupted_files)

# Print the list of non-JPG files
if all_non_jpg_files:
    print("Non-JPG files found:")
    for file in all_non_jpg_files:
        print(file)
else:
    print("No non-JPG files found in the specified directories.")

# Print the list of corrupted JPG files
if corrupted_jpg_files:
    print("\nCorrupted JPG files found:")
    for file in corrupted_jpg_files:
        print(file)
else:
    print("\nNo corrupted JPG files found.")
