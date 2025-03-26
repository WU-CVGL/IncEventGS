import os
import sys
from PIL import Image

def convert_gray_to_rgb(basedir, folder_name):
    # Determine the source and destination folder paths
    source_folder = os.path.join(basedir, folder_name)
    dest_folder = os.path.join(basedir, folder_name + "_3ch")
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} does not exist")
        return
    
    # Create destination folder if it does not exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        # Ensure the file is an image
        try:
            with Image.open(file_path) as img:
                # If the image is grayscale, convert it to RGB
                if img.mode == 'L':
                    rgb_img = img.convert('RGB')
                    # Save the converted image in the destination folder
                    dest_path = os.path.join(dest_folder, filename)
                    rgb_img.save(dest_path)
                    # print(f"{filename} has been converted and saved to {dest_folder}")
                else:
                    print(f"{filename} is not a grayscale image, skipping")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Get the base directory and folder name from command line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <basedir> <folder_name>")
        sys.exit(1)
    
    basedir = sys.argv[1]
    folder_name = sys.argv[2]
    
    # Call the conversion function
    convert_gray_to_rgb(basedir, folder_name)
