import cv2
from DataHandling.Episode import EpisodeManager, Episode

# Define colors for visualization
PATH_COLOR = (0, 255, 0)  # Green for all paths
CHOSEN_PATH_COLOR = (0, 0, 255)  # Red for the chosen path

def visualize_paths(frame):
    """
    Draw bounding boxes around detected paths using Shapely polygon bounds format.
    Highlight the chosen path with a different color.
    """
    image = frame.image.copy()  # Work on a copy of the image
    detections = frame.data['paths']
    chosen_path_id = frame.data['pathId']

    for path_id, det_info in detections.items():
        bbox = det_info['bbox']  # Expecting bbox as (xmin, ymin, xmax, ymax)
        color = CHOSEN_PATH_COLOR if str(path_id) == str(chosen_path_id) else PATH_COLOR
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(image, f'Path {path_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return image

def main():
    episodeManager = EpisodeManager(mode="read", loadLocation="DatabaseLabelled")

    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()


        for frame, i in episode:
            # Visualize paths on the frame
            visualize_frame = visualize_paths(frame)

            # Display the frame
            cv2.imshow('Path Visualization', visualize_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    episode = episodeManager.nextEpisode()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()