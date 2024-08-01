






from DataHandling.Episode import EpisodeManager, Episode

import numpy as np
import cv2





def findFeatures(image, featureType = "AKAZE"):

    if featureType == "AKAZE":
        detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0002,
            nOctaves=8,
            nOctaveLayers=8,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
    elif featureType == "ORB":
        detector = cv2.ORB_create()
    elif featureType == "BRISK":
        detector = cv2.BRISK_create()
    elif featureType == "SIFT":
        detector = cv2.SIFT_create()

    elif featureType == "KAZE":
        detector = cv2.KAZE_create()
    else:
        raise ValueError("Invalid feature type")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    image = clahe.apply(image)

    #blur
    #image = cv2.GaussianBlur(image, (3, 3), 0)

    keypoints, descriptors = detector.detectAndCompute(image, None)









    return image, keypoints, descriptors

def doTestTransform():

    # Create a new episode manager
    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")

    # Load the episodes
    #episodeManager.loadEpisodes()



    episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()


    lastFrame = None

    for frame, index in episode:
        if index == 0:
            lastFrame = frame
            continue

        oldIndex = max(index-1, 0)
        lastFrame = episode[oldIndex]

        # Convert the frame to grayscale
        newImage = frame.image
        oldImage = lastFrame.image

        # Find the features
        

        imageNewGray, keypointsNew, descriptorsNew = findFeatures(newImage, featureType="AKAZE")
        imageOldGray, keypointsOld, descriptorsOld = findFeatures(oldImage, featureType="AKAZE")


        # Match the features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptorsOld, descriptorsNew)

        # Sort the matches based on distance
        matches = sorted(matches, key = lambda x:x.distance)


        best_matches = matches[:10]




        oldImage = oldImage.copy()
        newImage = newImage.copy()

        # Draw the best 10 matches with different colors
        for i, match in enumerate(best_matches):
            old_idx = match.queryIdx
            new_idx = match.trainIdx
            pt_old = tuple(map(int, keypointsOld[old_idx].pt))
            pt_new = tuple(map(int, keypointsNew[new_idx].pt))

            
            
            cv2.circle(oldImage, pt_old, 5, (0, 0, 255), -1)
            cv2.circle(newImage, pt_new, 5, (0, 255, 0), -1)





        # Draw the matches
        #image = cv2.drawMatches(imageOld, keypointsOld, imageNew, keypointsNew, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        
        pointsOld = np.float32([keypointsOld[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pointsNew = np.float32([keypointsNew[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute the homography matrix
        H, mask = cv2.findHomography(pointsOld, pointsNew, cv2.RANSAC, 5.0)

        # Warp the old image to align with the new image
        height, width = newImage.shape[:2]
        warpedOldImage = cv2.warpPerspective(oldImage, H, (width, height))

        # Create a larger canvas to accommodate both images
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the new image on the canvas
        canvas[:height, :width] = newImage

        # Blend the warped old image into the canvas
        mask = (warpedOldImage > 0).astype(np.uint8)  # Create a mask where the warped image has content
        blendedImage = cv2.addWeighted(canvas, 0.5, warpedOldImage, 0.5, 0)

        # Display the composite image
        cv2.imshow('Composite Image', blendedImage)

        # Draw the matches
        matched_image = cv2.drawMatches(oldImage, keypointsOld, newImage, keypointsNew, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', matched_image)

        


        lastFrame = frame
        # Wait for a key press, but if the key is 'q', then break
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


        
    
        















if __name__ == '__main__':
    doTestTransform()