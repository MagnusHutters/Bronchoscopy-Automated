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
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        cv2.putText(image, f'Path {path_id}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

def main():
    episodeManager = EpisodeManager(mode="read", loadLocation="DatabaseManualBendOffset")

    episodeManager.currentIndex=0
    doQuit = False
    while episodeManager.hasNextEpisode() and not doQuit:
        episode = episodeManager.nextEpisode()
        print(f"Episode {episodeManager.currentIndex}")


        index=0

        while not doQuit:
            # Visualize paths on the frame

            frame = episode[index]
            visualize_frame = visualize_paths(frame)

            #scale up for better visualization
            visualize_frame = cv2.resize(visualize_frame, (800, 800))

            action = frame.action["char_value"]

            charToAction = {"f": "forward", "b": "backward", "l": "left", "r": "right", "u": "up", "d": "down", "": "no action"}

            

            #display action on frame
            cv2.putText(visualize_frame, f'F: {index} - Action: {charToAction[action]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Frame", visualize_frame)


            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                doQuit = True
                break
            elif key == ord('n'):
                break
            #a & d for previous and next frame
            elif key == ord('a'):
                index = max(0, index-1)
            elif key == ord('d'):
                index = min(len(episode)-1, index+1)
            #enter to skip to next episode
            elif key == 13:
                break
                

            
            


        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    episode = episodeManager.endEpisode(discard=True)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()