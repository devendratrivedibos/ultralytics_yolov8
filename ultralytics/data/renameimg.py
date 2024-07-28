

import os

def rename_files_with_prefix(directory, prefix):
    # Define the extensions of the files you want to rename
    extensions = ('.json', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has one of the specified extensions
        if filename.endswith(extensions):
            # Construct the old file path
            old_file_path = os.path.join(directory, filename)
            # Construct the new file name
            new_filename = prefix + "_" + filename
            # Construct the new file path
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")

# Specify the directory and prefix
directory = "D:/CRRI_demo/89b23187-36ed-44e8-be9a-8a934988cc2b/a908b650-3bd8-4fa7-b67a-210dc64e6ddd/range_images"
prefix = "a90"

# Call the function
rename_files_with_prefix(directory, prefix)
