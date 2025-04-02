import os

# Path to the validation directory
validation_dir = r'C:\Users\USER\Desktop\PLANT\Validation'

# Number of total classes we want
total_classes = 10001

# Get the current number of class folders in the validation directory
existing_folders = [folder for folder in os.listdir(validation_dir) if os.path.isdir(os.path.join(validation_dir, folder))]
existing_count = len(existing_folders)

# Determine how many more folders we need to create
folders_to_create = total_classes - existing_count

# Create missing folders, starting with class1, class2, ...
for i in range(existing_count + 1, total_classes + 1):
    folder_name = f'class{i}'
    folder_path = os.path.join(validation_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f'Created folder: {folder_name}')

print(f"Finished creating {folders_to_create} additional folders to reach {total_classes} total.")
