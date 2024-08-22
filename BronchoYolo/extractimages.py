import os
import zipfile
import random
import argparse

def extract_random_images(source_folder, destination_folder, total_images_to_extract, prefix):
    # Create the output folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Collect all the matching files from all zip files along with their zip file names
    file_zip_pairs = []

    zip_files = [f for f in os.listdir(source_folder) if f.endswith('.zip')]

    for zip_file in zip_files:
        with zipfile.ZipFile(os.path.join(source_folder, zip_file), 'r') as zip_ref:
            # List all files in the zip file
            all_files = zip_ref.namelist()

            # Filter files based on the prefix and store the pair (zip_file, file)
            prefix_files = [f for f in all_files if f.startswith(prefix)]
            file_zip_pairs.extend([(zip_file, f) for f in prefix_files])

    # Check if the total number of requested images is more than available
    if len(file_zip_pairs) < total_images_to_extract:
        print(f"Requested {total_images_to_extract} images, but only found {len(file_zip_pairs)}. Extracting all available images.")
        total_images_to_extract = len(file_zip_pairs)

    # Select random pairs from the list
    selected_pairs = random.sample(file_zip_pairs, total_images_to_extract)

    # Remap selected pairs into a dict with zip files as keys
    files_to_extract = {}
    for zip_file, file in selected_pairs:
        if zip_file not in files_to_extract:
            files_to_extract[zip_file] = []
        files_to_extract[zip_file].append(file)

    # Extract the selected files from their respective zip files, renaming them to avoid conflicts
    for zip_file, files in files_to_extract.items():
        with zipfile.ZipFile(os.path.join(source_folder, zip_file), 'r') as zip_ref:
            for file in files:
                # Extract the file
                extracted_path = zip_ref.extract(file, destination_folder)
                # Create a new name for the file with the zip file name as a prefix
                new_name = f"{os.path.splitext(zip_file)[0]}_{os.path.basename(file)}"
                new_path = os.path.join(destination_folder, new_name)

                # Check if the file already exists
                if os.path.exists(new_path):
                    print(f"File {new_name} already exists. Skipping...")
                    # Remove the extracted file
                    os.remove(extracted_path)

                else:
                # Rename the file to avoid conflicts
                    os.rename(extracted_path, new_path)

    print("Extraction and renaming complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a total number of random images from all zip files.')

    parser.add_argument('source_folder', type=str, help='The folder containing the zip files.')
    parser.add_argument('destination_folder', type=str, help='The folder where extracted images will be saved.')
    parser.add_argument('total_images_to_extract', type=int, help='Total number of images to extract from all zip files.')
    parser.add_argument('prefix', type=str, help='The prefix for the images to be extracted.')

    args = parser.parse_args()

    # Check if the number of images to extract is positive
    if args.total_images_to_extract <= 0:
        parser.error("The total number of images to extract must be a positive integer.")

    extract_random_images(args.source_folder, args.destination_folder, args.total_images_to_extract, args.prefix)
