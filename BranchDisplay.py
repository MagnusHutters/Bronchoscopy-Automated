


from branchModel import BranchModel

from DataHandling.Episode import Episode, EpisodeManager

import cv2



def displayBranches():
    
    episodeManager = EpisodeManager(mode = "Read", loadLocation="Database")

    episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    model = BranchModel("C:/Users/magnu/OneDrive/Misc/Ny mappe/Bronchoscopy-Automated/BranchModel.h5")

    currentIndex = 150

    while True:

        frame = episode[currentIndex]

        image = frame.image.copy()

        correctPrediction, wrongPredictions, allPredictions = model.predict(image)




        for prediction in allPredictions:

            #color based on prediction.certainty - certainty is a value between 0 and 1 where 1 is most certain
            #green is most certain, red is least certain

            certainty = prediction.certainty
            
            color = (0,int(255 * certainty), int(255 * (1 - certainty)))

            contour = prediction.contour
            #print(contour)
            cv2.drawContours(image, [contour], -1, color, 1)

            center = prediction.point

            cv2.circle(image, center, 3, color, -1)

            #draw text
            text = f"{float(prediction.certainty):.2f}"
            cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow("Branches", image)

        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord('a'):
            currentIndex -= 1
        elif key == ord('d'):
            currentIndex += 1



        #make sure index is within bounds
        currentIndex = max(0, min(len(episode) - 1, currentIndex))
            

            






if __name__ == "__main__":
    displayBranches()





