import os
import glob

def clean_up_specific_files(directory):
    """
    Removes specific files and files matching a pattern within the specified directory.
    
    :param directory: The path to the directory where files will be removed.
    """
    # Files and patterns to delete
    files_and_patterns = [
        "helium_abundances.png",
        "detonation_lengths.png",
        "reaction_flow_*.png"  # This will match all files starting with 'reaction_flow_' and ending with '.png'
    ]
    
    # Iterate over each file/pattern to delete them
    for item in files_and_patterns:
        full_path_pattern = os.path.join(directory, item)
        
        # For patterns, glob.glob will find matches. For specific files, it will return the file if it exists.
        files_to_delete = glob.glob(full_path_pattern)
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Use the current working directory as the directory to clean
current_directory = os.getcwd()

clean_up_specific_files(current_directory)

