import os
import shutil

def recursive_mkdir(path):
    # Split the path into its components
    parts = path.split(os.sep)
    
    # Rebuild the path step by step
    current_path = ""
    
    for part in parts:
        # Append the current part to the path
        current_path = os.path.join(current_path, part)
        
        # If the current path doesn't exist, create it
        if not os.path.exists(current_path):
            os.makedirs(current_path)
            print(f"Created directory: {current_path}")

def find_and_copy_file(source_folder, dest_folder, filename):
    # Iterate through all files in the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # Check if the file matches the desired filename
            if filename in file:
                # Construct the full path of the file
                source_file = os.path.join(root, file)
                
                # Construct the destination file path
                dest_file = os.path.join(dest_folder, file)
                
                # Copy the file to the destination folder
                shutil.copy(source_file, dest_file)
                print(f"Copied {file} to {dest_folder}")
                return  # Stop after the first match (if you only want one match)
    
    print(f"No file named '{filename}' found in {source_folder}.")