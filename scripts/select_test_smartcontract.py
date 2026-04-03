import os
import random
import shutil

def copy_random_contracts(source_dir, dest_dir, num_files=50):
    """
    Randomly select and copy .sol files from source_dir to dest_dir.

    Args:
        source_dir (str): Source directory containing .sol files.
        dest_dir (str): Destination directory for copied files.
        num_files (int): Number of files to copy (default: 50).
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of .sol files in source directory
    sol_files = [f for f in os.listdir(source_dir) if f.endswith('.sol')]
    
    if not sol_files:
        raise ValueError(f"No .sol files found in {source_dir}")
    
    if len(sol_files) < num_files:
        raise ValueError(f"Requested {num_files} files, but only {len(sol_files)} .sol files available in {source_dir}")

    # Randomly select num_files
    selected_files = random.sample(sol_files, num_files)
    
    # Copy selected files to destination directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy2(source_path, dest_path)
        print(f"Copied {file_name} to {dest_dir}")

if __name__ == "__main__":
    # Default: use paths relative to this script's location
    base_dir = os.path.join(os.path.dirname(__file__), "..", "test_smartcontract_dataset")
    source_dir = os.path.join(base_dir, "smartbugs-wild", "contracts")
    dest_dir = os.path.join(base_dir, "test_dataset_for_train")

    try:
        copy_random_contracts(source_dir, dest_dir, num_files=1)
        print(f"Successfully copied .sol files to {dest_dir}")
    except Exception as e:
        print(f"Error: {str(e)}")
