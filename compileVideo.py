import cv2
import os
import glob
import sys
import zipfile
import tempfile
import shutil




def create_video_from_images(image_folder, video_name, fps=20):
    images = sorted(glob.glob(os.path.join(image_folder, f"{video_name}*.png")))

    if not images:
        print(f"No images found for {video_name} in {image_folder}")
        return

    print(f"Rendering {video_name}.mp4")

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(image_folder, f'00_{video_name}.mp4')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    total_images = len(images)

    for idx, image in enumerate(images):
        video.write(cv2.imread(image))
        progress = (idx + 1) / total_images * 100
        print(f"\rCreating {video_name}.mp4: frame {idx + 1} of {total_images} ({progress:.2f}%)", end='')

    video.release()
    print(f"\nVideo {video_name}.mp4 created successfully at {video_path}.")

def resize_to_same_height(image1, image2):
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    if height1 != height2:
        if height1 > height2:
            new_width2 = int(width2 * (height1 / height2))
            image2 = cv2.resize(image2, (new_width2, height1), interpolation=cv2.INTER_LINEAR)
        else:
            new_width1 = int(width1 * (height2 / height1))
            image1 = cv2.resize(image1, (new_width1, height2), interpolation=cv2.INTER_LINEAR)

    return image1, image2

def create_combined_video(image_folder, fps=20):
    broncho_images = sorted(glob.glob(os.path.join(image_folder, "broncho*.png")))
    top_images = sorted(glob.glob(os.path.join(image_folder, "top*.png")))

    if not broncho_images or not top_images:
        print("No images found for broncho or top in the specified folder.")
        return

    print("Rendering combined.mp4")

    broncho_frame = cv2.imread(broncho_images[0])
    top_frame = cv2.imread(top_images[0])

    broncho_frame, top_frame = resize_to_same_height(broncho_frame, top_frame)

    height = broncho_frame.shape[0]
    width = broncho_frame.shape[1] + top_frame.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(image_folder, '00_combined.mp4')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    total_images = min(len(broncho_images), len(top_images))

    for idx in range(total_images):
        broncho_frame = cv2.imread(broncho_images[idx])
        top_frame = cv2.imread(top_images[idx])

        broncho_frame, top_frame = resize_to_same_height(broncho_frame, top_frame)
        combined_frame = cv2.hconcat([broncho_frame, top_frame])
        video.write(combined_frame)
        progress = (idx + 1) / total_images * 100
        print(f"\rCreating combined.mp4: frame {idx + 1} of {total_images} ({progress:.2f}%)", end='')

    video.release()
    print(f"\nVideo combined.mp4 created successfully at {video_path}.")

def create_videos_from_folder(folder_path, create_combined=False):
    if folder_path.endswith('.zip'):
        print(f"Extracting images from {folder_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(folder_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            create_video_from_images(temp_dir, 'broncho', fps=20)
            create_video_from_images(temp_dir, 'top', fps=20)
            if create_combined:
                create_combined_video(temp_dir, fps=20)
            # Copy videos back to the original folder location
            video_files = glob.glob(os.path.join(temp_dir, '00_*.mp4'))
            for video_file in video_files:
                shutil.copy(video_file, os.path.dirname(folder_path))
            # Add the videos to the zip file
            print(f"Repacking videos to {folder_path}")
            with zipfile.ZipFile(folder_path, 'a') as zip_ref:
                for video_file in video_files:
                    zip_ref.write(video_file, os.path.basename(video_file))
    else:
        #create_video_from_images(folder_path, 'broncho', fps=20)
        #create_video_from_images(folder_path, 'top', fps=20)
        if create_combined:
            create_combined_video(folder_path, fps=20)

    print("Done Rendering.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_path_or_zip> [--combine]")
        sys.exit(1)

    folder_path = sys.argv[1]
    create_combined = '--combine' in sys.argv

    create_videos_from_folder(folder_path, create_combined=create_combined)
