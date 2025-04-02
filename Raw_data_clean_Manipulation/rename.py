import os
import xml.etree.ElementTree as ET
import re

# Define the directory containing the folders
base_dir = r"C:\Users\USER\Desktop\PLANT\PlantCLEF2017Train1EOL\dificult to rename"

def sanitize_xml(xml_content):
    """
    Sanitizes the XML content to handle unescaped characters like '&'.
    Replaces them with their corresponding escape sequences.
    """
    # Replace any unescaped '&' with '&amp;', but only if it's not already part of an escaped entity
    sanitized_content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', xml_content)
    return sanitized_content

def extract_labels_from_xml(xml_file):
    """
    Extracts Species, Genus, and Family from the sanitized XML content.
    """
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Sanitize the XML content
    xml_content = sanitize_xml(xml_content)

    # Parse the sanitized XML content
    root = ET.fromstring(xml_content)

    species = root.find('Species').text.strip().replace(" ", "_")
    genus = root.find('Genus').text.strip().replace(" ", "_")
    family = root.find('Family').text.strip().replace(" ", "_")

    return species, genus, family

def rename_folders(base_dir):
    """
    Renames folders based on the first .xml file found in each folder.
    """
    # Loop through all the folders in the base directory
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            # Look for the first .xml file in the folder
            xml_file = None
            for file in os.listdir(folder_path):
                if file.endswith(".xml"):
                    xml_file = os.path.join(folder_path, file)
                    break  # Stop after finding the first XML file

            if xml_file:
                try:
                    species, genus, family = extract_labels_from_xml(xml_file)
                    new_folder_name = f"{species}_{genus}_{family}"
                    new_folder_path = os.path.join(base_dir, new_folder_name)

                    # Rename the folder
                    os.rename(folder_path, new_folder_path)
                    print(f"Renamed '{folder}' to '{new_folder_name}'")
                except Exception as e:
                    print(f"Error processing {xml_file}: {e}")

if __name__ == "__main__":
    rename_folders(base_dir)
