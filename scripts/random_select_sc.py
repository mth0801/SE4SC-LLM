import os
import shutil
import random
import logging

def copy_random_files(source_dir, dest_dir, num_files=20):
    """
    Randomly select and copy a specified number of files from source to destination directory.
    
    Args:
        source_dir (str): Path to source directory
        dest_dir (str): Path to destination directory
        num_files (int): Number of files to copy (default: 50)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate source directory
    if not os.path.isdir(source_dir):
        logger.error(f"Source directory '{source_dir}' does not exist")
        raise ValueError("Source directory does not exist")
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get list of all files in source directory (excluding subdirectories)
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Check if there are enough files
    if len(files) < num_files:
        logger.error(f"Source directory has only {len(files)} files, need {num_files}")
        raise ValueError(f"Not enough files in source directory")
    
    # Randomly select files
    selected_files = random.sample(files, num_files)
    
    # Copy selected files to destination
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        
        # Avoid overwriting by appending a number if file exists
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(dest_path):
            new_file_name = f"{base}_{counter}{ext}"
            dest_path = os.path.join(dest_dir, new_file_name)
            counter += 1
        
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied '{file_name}' to '{dest_path}'")
        except Exception as e:
            logger.error(f"Failed to copy '{file_name}': {str(e)}")
            raise
    
    logger.info(f"Successfully copied {num_files} files to '{dest_dir}'")

if __name__ == "__main__":
    # Use relative paths from the script directory
    script_dir = os.path.dirname(__file__)
    source_directory = os.path.join(script_dir, "..", "test_smartcontract_dataset", "smartbugs-wild", "contracts")
    destination_directory = os.path.join(script_dir, "..", "test_smartcontract_dataset", "dataset_for_train")
    
    try:
        copy_random_files(source_directory, destination_directory, num_files=50)
    except Exception as e:
        logging.error(f"Error: {str(e)}")