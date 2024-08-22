

import os
import shutil
import random
import argparse
from pathlib import Path

def move_files(source_folder, destination_folder, num_files):
    # Ensure the destination folder exists
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Get a list of all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Shuffle the files randomly (optional)
    random.shuffle(files)

    # Select the specified number of files
    files_to_move = files[:num_files]

    # Move each selected file to the destination folder
    for file_name in files_to_move:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        shutil.move(source_path, destination_path)

    print(f"Moved {len(files_to_move)} files from {source_folder} to {destination_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move a specified number of files from one folder to another.")
    parser.add_argument('source_folder', type=str, help='Path to the source folder.')
    parser.add_argument('destination_folder', type=str, help='Path to the destination folder.')
    parser.add_argument('num_files', type=int, help='Number of files to move.')

    args = parser.parse_args()

    move_files(args.source_folder, args.destination_folder, args.num_files)
