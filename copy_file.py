import shutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def copy_file(source_file, target_file):
    try:
        shutil.copy(source_file, target_file)
    except Exception as e:
        print(f"Error copying {source_file}: {e}")

def copy_all_files_concurrently(source_base, target_base, section_numbers, subfolder_name):
    file_paths = []

    # Gather all file paths
    for section_number in section_numbers:
        source_dir = os.path.join(source_base, f"SECTION-{section_number}", subfolder_name)
        if os.path.exists(source_dir):
            os.makedirs(os.path.join(target_base, f"SECTION-{section_number}", subfolder_name), exist_ok=True)
            files = os.listdir(source_dir)
            for file in files:
                source_file = os.path.join(source_dir, file)
                target_file = os.path.join(target_base, f"SECTION-{section_number}", subfolder_name, file)
                file_paths.append((source_file, target_file))
        else:
            print(f"Source directory {source_dir} does not exist")

    # Copy files concurrently
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(copy_file, source_file, target_file)
            for source_file, target_file in file_paths
        ]
        # Wait for all futures to complete
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions that occurred


source_base = r"Z:/ALIGARH-KANPUR_2024-09-21_11-50-15"
target_base = r"D:/ALIGARH-KANPUR_2024-09-21_11-50-15"
section_numbers = range(4, 6)

copy_all_files_concurrently(source_base, target_base, section_numbers, "r3")
print("All r3 copied successfully.")

copy_all_files_concurrently(source_base, target_base, section_numbers, "insd")
print("All insd copied successfully.")
