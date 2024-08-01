

from DataHandling.Episode import EpisodeManager, Episode

import numpy as np
import cv2
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, maximum_filter

import matplotlib.pyplot as plt



mask = None


def createVignetteMask(image):
    rows, cols = image.shape


    radius_ratio=1
    strength = 0.75

    # Generate vignette mask using Gaussian kernels
    center_x, center_y = cols // 2, rows // 2
    radius = min(center_x, center_y) * radius_ratio
    mask = np.zeros((rows, cols), np.float32)

    for y in range(rows):            
        for x in range(cols):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if distance <= radius:
                mask[y, x] = 1
            else:

                distanceOuside = distance - radius

                fallOff = strength * radius

                mask[y, x] = max(0.0, 1.0 - (distanceOuside / fallOff))


                #mask[y, x] = 0.0
    return mask


def preProcessImage(image):
    global mask
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    #image = clahe.apply(image)

    #blur
    image = cv2.GaussianBlur(image, (11, 11), 0)
    

    #invert the image
    image = cv2.bitwise_not(image)

    if mask is None:
        mask = createVignetteMask(image)

    


    cv2.imshow("Mask", mask)

    # Apply the mask to the image
    vignette = image * mask
    

    # Normalize the image
    vignette = cv2.normalize(vignette, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    vignette = cv2.bitwise_not(vignette)

    return vignette


def findBranches(originalImage):
    

    image = preProcessImage(originalImage)
    
    #display the image
    cv2.imshow("Preprocessed Image", image)

    points = []



    for threshold in range(1,128):


        depth = 128-threshold
        #inverse binary threshold
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

        #find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #draw contours
        for contour in contours:
            
            #get the moments
            M = cv2.moments(contour)
            area = cv2.contourArea(contour)
            circumference = cv2.arcLength(contour, True)

            roundness = 0
            if circumference == 0:
                roundness = 0
            else:
                roundness = 4 * np.pi * area / (circumference * circumference)




            if area < 5:
                continue
            if area > 10000:
                continue

            #calculate the center of mass
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append((cX, cY, roundness, area, circumference, depth))


    drawImage = originalImage.copy()
    #draw the centers

    image = image.astype(np.float32)
    


    



    density = np.zeros_like(image)
    for x, y, roundness, area, circumference, depth in points:
        print(f"X: {x}, Y: {y}, Roundness: {roundness}, Area: {area}, Circumference: {circumference}")
        density[y, x] += 100+(roundness*area*depth)


    density = gaussian_filter(density, 15.0)
    density = cv2.sqrt(density)

    density = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    neighborhood_size = 15
    local_max = maximum_filter(density, size=neighborhood_size) == density

    threshold = 64
    detected_peaks = (density > threshold) & local_max

    peak_coords = np.argwhere(detected_peaks)




    cv2.imshow("Density", density)


    for point in points:
        cv2.circle(drawImage, (point[0], point[1]), 5, (0, 0, 255), -1)


    for peak in peak_coords:
        cv2.circle(drawImage, (peak[1], peak[0]), 5, (0, 255, 0), -1)
    
    cv2.imshow("Image", drawImage)






if __name__ == '__main__':
    


    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")


    episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    for frame, index in episode:
        print(frame.image.shape)
        findBranches(frame.image)

        
        key = cv2.waitKey(0)
        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
        # if esc is pressed, exit the loop
        elif key == 27:
            break

    cv2.destroyAllWindows()

