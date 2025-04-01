import os

# Define the directory path
dir_path = r'C:\Users\USER\Desktop\PLANT\test'

# Walk through the directory and its subdirectories
for foldername, subfolders, filenames in os.walk(dir_path):
    for filename in filenames:
        # Check if the file ends with .xml
        if filename.endswith('.xml'):
            file_path = os.path.join(foldername, filename)
            try:
                # Delete the .xml file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

print("All .xml files have been deleted.")
