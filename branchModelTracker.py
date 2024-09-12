
import copy
import numpy as np
import cv2
import torch
import time
import os
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

from math import sqrt, pi

from scipy.optimize import linear_sum_assignment

from DataHandling.Episode import EpisodeManager, Episode

import time
import warnings


from Timer import Timer

import warnings





def detect_features(image, feature_type = "AKAZE"):
        
    """
    Detects features in an image based on the specified feature type.
    
    Parameters:
    image (numpy.ndarray): The input image.
    feature_type (str): The type of feature to detect ('SIFT', 'ORB', etc.)
    
    Returns:
    keypoints (list): List of detected keypoints.
    descriptors (numpy.ndarray): Corresponding descriptors for the keypoints.
    """


    if feature_type == "AKAZE":
        detector = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=0.0001,
            nOctaves=8,
            nOctaveLayers=8,
            diffusivity=cv2.KAZE_DIFF_PM_G2
        )
    elif feature_type == "ORB":
        detector = cv2.ORB_create()
    elif feature_type == "BRISK":
        detector = cv2.BRISK_create()
    elif feature_type == "SIFT":
        detector = cv2.SIFT_create()

    elif feature_type == "KAZE":
        detector = cv2.KAZE_create()
    else:
        raise ValueError("Invalid feature type")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    image = clahe.apply(image)

    #blur
    #image = cv2.GaussianBlur(image, (3, 3), 0)

    cv2.imshow("processedImage", image)

    keypoints, descriptors = detector.detectAndCompute(image, None)









    return keypoints, descriptors


def find_affine_transformation(img2, oldKeypoints, oldDescriptors, feature_type='AKAZE'):
    """
    Finds the affine transformation between two images by detecting and matching features.
    
    Parameters:
    img1 (numpy.ndarray): The first input image.
    img2 (numpy.ndarray): The second input image.
    feature_type (str): The type of feature to detect and match ('SIFT', 'ORB', etc.)
    
    Returns:
    numpy.ndarray: The affine transformation matrix.
    """

    #if img1 is None return a affine transformation matrix that does nothing

    doDownsample = True
    downSampleFactor = 0.5

    if doDownsample:
        img2 = cv2.resize(img2, (0, 0), fx=downSampleFactor, fy=downSampleFactor)

        
    


    # Detect features in both images
    #keypoints1, descriptors1 = detect_features(img1, feature_type)
    newKeypoints, newDescriptors = detect_features(img2, feature_type)

    if doDownsample:
        #upsample the keypoints to the original size

        for keypoint in newKeypoints:
            keypoint.pt = (keypoint.pt[0]/downSampleFactor, keypoint.pt[1]/downSampleFactor)

    if oldKeypoints is None or oldDescriptors is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), newKeypoints, newDescriptors
    
    # Match features between the two images
    if feature_type == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif feature_type == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif feature_type == 'AKAZE':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(oldDescriptors, newDescriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    src_pts = np.float32([oldKeypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([newKeypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    #print(f"Number of matches: {len(matches)}")


    #display the matches
    #img_matches = cv2.drawMatches(img1, oldKeypoints, img2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow("Matches", img_matches)


    # Compute the affine transformation matrix
    affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if affine_matrix is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), newKeypoints, newDescriptors
    


    overallRotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])

    overallScaleX = sqrt(affine_matrix[0, 0] ** 2 + affine_matrix[1, 0] ** 2)

    overallScaleY = sqrt(affine_matrix[0, 1] ** 2 + affine_matrix[1, 1] ** 2)

    maxScale = 2
    minScale = 0.5
    maxRotation = pi/4



    tooLarge = False

    if max(overallScaleX, overallScaleY) > maxScale:
        tooLarge = True

    if min(overallScaleX, overallScaleY) < minScale:
        tooLarge = True


    if abs(overallRotation) > maxRotation:
        tooLarge = True

    if len(matches) < 25:
        tooLarge = True

    if tooLarge:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), newKeypoints, newDescriptors



    
    return affine_matrix, newKeypoints, newDescriptors



def create_detections_from_yolo(yolo_predictions):
    """
    Converts raw YOLO detections into a dictionary of Detection objects, grouped by class.
    
    Parameters:
    yolo_detections (list): A list of YOLO detections, where each detection is a tuple 
                            (class_id, confidence, x_min, y_min, x_max, y_max)
    
    Returns:
    dict: A dictionary with class_id as keys and lists of Detection objects as values.
    """
    detections = []

    predictions = yolo_predictions.pred[0]



    for yoloDetection in predictions:


        detection = Detection.from_yolo_detection(yoloDetection)

        detections.append(detection)
    
    return detections

def transform_all_detections(detections, affine_matrix):
    """
    Applies an affine transformation to all detections in a dictionary of lists.

    Parameters:
    detections_dict (dict): A dictionary with class_id as keys and lists of Detection objects as values.
    affine_matrix (numpy.ndarray): The affine transformation matrix.

    Returns:
    dict: A new dictionary with transformed Detection objects.
    """
    #print(f"affine_matrix: {affine_matrix}")


    #transformed_detections_list = []
    for detection in detections:
        detection.apply_affine_transformation(affine_matrix)
        #transformed_detections_list.append(detection)

    #return transformed_detections_dict




def match_detections(list1, list2, min_similarity=0.1):
    """
    Matches detections between two lists using the Hungarian algorithm.

    Parameters:
    list1 (list): A list of Detection objects.
    list2 (list): Another list of Detection objects.
    min_similarity (float): The minimum similarity score required to consider a match. Default is 0.5.

    Returns:
    list: A list of tuples. Each tuple contains a matched Detection object from list1 and list2, 
          or (item, None) if the detection in list1 has no suitable match in list2.
    """
    num_detections_1 = len(list1)
    num_detections_2 = len(list2)

    #print(f"num_detections_1: {num_detections_1}")
    #print(f"num_detections_2: {num_detections_2}")

    #handle the case where one of the lists is empty
    if num_detections_1 == 0:
        return [(None, detection) for detection in list2]
    
    if num_detections_2 == 0:
        return [(detection, None) for detection in list1]



    # Create a cost matrix where each entry is the negative similarity score (since the algorithm minimizes cost)
    cost_matrix = np.zeros((num_detections_1, num_detections_2))

    for i, det1 in enumerate(list1):
        for j, det2 in enumerate(list2):
            similarity = det1.get_similarity(det2)
            cost_matrix[i, j] = -similarity if similarity >= min_similarity else 999999


    #print(cost_matrix)
    # Solve the assignment problem using the Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Extract the matched pairs based on the indices returned by the Hungarian algorithm
    matched_pairs = []
    matched_rows = set()
    matched_cols = set()

    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < 1000:  # This ensures the match meets the minimum similarity threshold
            matched_pairs.append((list1[row], list2[col]))
            matched_rows.add(row)
            matched_cols.add(col)

    # Add unmatched detections from list1
    for i in range(num_detections_1):
        if i not in matched_rows:
            matched_pairs.append((list1[i], None))

    for j in range(num_detections_2):
        if j not in matched_cols:
            matched_pairs.append((None, list2[j]))

    return matched_pairs






def buildHierachy(detections):

    for detection in detections:
        detection.children = []
        detection.parent = None
    

    for firstDetection in detections:
        for secondDetection in detections:
            if firstDetection is secondDetection:
                continue

            if firstDetection.contains(secondDetection):
                firstDetection.children.append(secondDetection)
                secondDetection.parent = firstDetection



class Detection:

    
    imageSize=(400,400)

    margin = 20
    
    viewPolygon = Polygon([(margin, margin), (margin, imageSize[1]-margin), (imageSize[0]-margin, imageSize[1]-margin), (imageSize[0]-margin, margin)])





    def __init__(self, class_id, confidence, polygon, id , bbox=None):
        """
        Initializes a Detection object with the actual values.

        Parameters:
        class_id (int): The class ID of the detection.
        confidence (float): The confidence score of the detection.
        polygon (Polygon): A shapely polygon object representing the detection's area.
        bbox (tuple): The bounding box of the detection (optional).
        """



        self.children=[]
        self.parent=None
        
        self.id=id
        self.class_id = int(class_id)
        self.confidence = float(confidence)
        self.polygon = polygon
        self.bbox = bbox if bbox is not None else self.polygon.bounds



        self.inView = False
        self.intersectionWithView = Polygon()


    def toDict(self):
        #return a json serializable dictionary of the detection
        return {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "polygon": list(self.polygon.exterior.coords),
            "bbox": self.bbox,
            "id": self.id,
            "inView": self.inView,
            "parent": self.parent.id if self.parent is not None else -1,
            "children": [child.id for child in self.children]
        }

    @classmethod
    def from_yolo_detection(cls, yolo_detection):
        """
        Class method to initialize a Detection object from YOLO detection data.

        Parameters:
        yolo_detection (tuple): A tuple containing (x_min, y_min, x_max, y_max, confidence, class_id).
        
        Returns:
        Detection: An initialized Detection object.
        """

        bbox = yolo_detection[:4]
        confidence = yolo_detection[4]
        class_id = yolo_detection[5]


        polygon = cls.create_polygon_from_bbox(bbox)
        return cls(class_id, confidence, polygon, bbox, None)

    @staticmethod
    def create_polygon_from_bbox(bbox):
        """
        Creates a polygon from the bounding box.

        Parameters:
        bbox (tuple): A tuple containing (x_min, y_min, x_max, y_max)
        
        Returns:
        Polygon: A shapely polygon object representing the bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        return Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])
    
        self.calculateViewIntersection()

    def apply_affine_transformation(self, affine_matrix):
        """
        Applies an affine transformation to the polygon.

        Parameters:
        affine_matrix (numpy.ndarray): The affine transformation matrix.
        """
        # Shapely affine_transform expects a 6-tuple (a, b, d, e, xoff, yoff)
        a, b, d, e = affine_matrix[0, 0], affine_matrix[0, 1], affine_matrix[1, 0], affine_matrix[1, 1]
        xoff, yoff = affine_matrix[0, 2], affine_matrix[1, 2]
        affine_params = (a, b, d, e, xoff, yoff)
        self.polygon = affine_transform(self.polygon, affine_params)
        # Update bounding box after transformation
        self.bbox = self.polygon.bounds


    def calculateViewIntersection(self):
        self.intersectionWithView = self.polygon.intersection(self.viewPolygon)

        self.inView = self.intersectionWithView.area > 1

    def apply_match(self, matched_detection, interpolation_factor=0.5):
        """
        Applies the result of a match to this detection, updating the polygon and confidence.

        Parameters:
        matched_detection (Detection or None): The matched detection object, or None if no match.
        """



        self.calculateViewIntersection()

        if matched_detection is not None:
            # Update polygon (for simplicity, we'll just keep the current polygon, but this could be extended)
            # Update confidence by adding the matched detection's confidence
            self.confidence += matched_detection.confidence


            self.polygon = matched_detection.polygon
            self.bbox = matched_detection.bbox
            self.confidence *= 0.98



        else:
            # Decrease confidence by a fixed amount or percentage if no match
            

            

            if self.inView:
            
                self.confidence *= 0.9

                if self.confidence < 0.1:
                    #remove the detection

                    self.confidence = 0


            else:
                self.confidence *= 0.94

                if self.confidence < 0.1:
                    #remove the detection

                    self.confidence = 0

            


                                                             

    def isInsided(self, detection): #check if the other detection is inside this detection, counts as inside if the intersection is more than 80% of this detection and not the opposite
        intersection = self.polygon.intersection(detection.polygon)

        insideCriteria=0.7
        return intersection.area > insideCriteria * self.polygon.area  and intersection.area < insideCriteria * detection.polygon.area

    def contains(self, detection): #check if this detection contains the other detection, counts as inside if the intersection is more than 80% of the other detection

        return detection.isInsided(self)      

    def get_intersection(self, other_detection):
        """
        Computes the intersection area with another detection.

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The intersection area.
        """
        return self.polygon.intersection(other_detection.polygon).area

    def get_union(self, other_detection):
        """
        Computes the union area with another detection.

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The union area.
        """
        return self.polygon.union(other_detection.polygon).area

    def get_similarity(self, other_detection):
        """
        Computes the similarity score with another detection using Intersection over Union (IoU).

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The similarity score (IoU).
        """
        intersection_area = self.get_intersection(other_detection)
        union_area = self.get_union(other_detection)
        return intersection_area / union_area if union_area != 0 else 0

class BranchModelTracker:
    def __init__(self, modelPath):
        pass

        self.model = torch.hub.load('BronchoYolo/yolov5', 'custom', path=modelPath, source='local', force_reload=True)
        self.model.eval()


        self.oldImage = None
        self.newId = 0


        self.trackedDetections = None

        self.oldSescriptors = None
        self.oldKeypoints = None
    

    def detectionsToPoints(self, detections):
        #takes the center of the bounding box of the detections and returns them as a list of points

        buildHierachy(detections)



        finalDetections={}
        points = {}
        if detections is not None:
            
            #add point if: 
                # the detection has no children and more than one sibling 
                # the detection has exactly one child
                # the detection has not parent or children


            for detection in detections:

                numSiblings =0
                if detection.parent is not None:
                    numSiblings = len(detection.parent.children)-1

                numChildren = len(detection.children)
                doAdd=False

                if numSiblings > 0 and numChildren == 0:
                    doAdd=True
                elif numChildren == 1:
                    doAdd=True
                elif detection.parent is None and numChildren == 0:
                    doAdd=True

                doAdd=True

                
                if doAdd:
                    if detection.inView:
                            
                        point = detection.polygon.centroid
                        points[detection.id] = (point.x, point.y)
                    else:
                        viewSize = Detection.imageSize

                        centerX, centerY = viewSize[1]//2, viewSize[0]//2
                        radius = min(viewSize)//2
                        radius -= Detection.margin

                        detectionPoint = detection.polygon.centroid

                        closestPoint = closest_point_on_circle(centerX, centerY, radius, detectionPoint.x, detectionPoint.y)

                        points[detection.id] = (int(closestPoint[0]), int(closestPoint[1]))

                    finalDetections[detection.id] = copy.deepcopy(detection)
                


        return points, finalDetections





    def getPredictions(self, image):

        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
        
            results = self.model(image)

        return results

    def apply_matches(self,matches, trackedDetections):
        """
        Applies matches to a list of detections.

        Parameters:
        matches (list): A list of tuples, where each tuple contains a detection from list1 and its matched detection
                        from list2 or None if no match was found.
        """
        for det1, det2 in matches:
            if det1 is not None:
                det1.apply_match(det2)
            else:
                #add the unmatched new detection to the list of tracked detections
                det2.id = self.newId
                self.newId += 1
                trackedDetections.append(det2)

                
        trackedDetections = [detection for detection in trackedDetections if detection.confidence > 0]


        return trackedDetections

    def predict(self, image, doDebug=False, doVideo=False, videoWriter=None):

        report=Timer.reset()
        #print(report)
        Timer.point("predict")



        newImage = image
        oldImage = self.oldImage

        self.oldImage = newImage

        

        Timer.point("getPredictions")


        time1 = time.time()
        newPredictions = self.getPredictions(newImage)
        time2 = time.time()
        #print(f"YOLO inference time: {time2-time1} seconds")
        

        

        #print(newPredictions)

        Timer.point("createDetections")
        newDetections = create_detections_from_yolo(newPredictions)

        #draw raw predictions
        newDetectionsImage = newImage.copy()

        if doDebug:
            for detection in newDetections:
                cv2.polylines(newDetectionsImage, [np.array(detection.polygon.exterior.coords, np.int32)], True, (0, 0, 255), 1)

                #draw the confidence
                cv2.putText(newDetectionsImage, f"{detection.confidence:.2f}", (int(detection.bbox[0]), int(detection.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("New Detections", newDetectionsImage)
            

        if self.trackedDetections is None:
            self.trackedDetections = newDetections

            for detection in self.trackedDetections:
                detection.id = self.newId
                self.newId += 1

            return self.detectionsToPoints(self.trackedDetections)


        Timer.point("findAffineTransformation")

        time3 = time.time()

        
        #affineTransformation = find_affine_transformation(oldImage, newImage, feature_type='AKAZE')
        affineTransformation, self.oldKeypoints, self.oldSescriptors = find_affine_transformation(newImage, self.oldKeypoints, self.oldSescriptors, feature_type='AKAZE')


        
        time4 = time.time()
        #print(f"Affine transformation time: {time4-time3} seconds")


        #display the affine transformation matrix: transform the old image and overlay it on the new image
        Timer.point("displayAffineTransformation")

        if doDebug:
            transformedOldImage = cv2.warpAffine(oldImage, affineTransformation, (newImage.shape[1], newImage.shape[0]))
            combinedImage = cv2.addWeighted(newImage, 0.5, transformedOldImage, 0.5, 0)

            #stacke 3 images horizontally
            combinedImage = np.hstack((newImage, combinedImage, transformedOldImage))

            cv2.imshow("Combined Image", combinedImage)


        #apply the affine transformation to the detections


        Timer.point("transformAllDetections")
        transform_all_detections(self.trackedDetections, affineTransformation)

        Timer.point("matchDetections")
        matchedPairs = match_detections(self.trackedDetections, newDetections)

        Timer.point("applyMatches")
        self.trackedDetections=self.apply_matches(matchedPairs,self.trackedDetections)


        
        Timer.point("Draw Detections")
        if doDebug or doVideo:
            drawImage = newImage.copy()
            for detection in self.trackedDetections:


                text = f"id: {detection.id} cfd: {detection.confidence:.2f}"
                if detection.inView:
                    cv2.polylines(drawImage, [np.array(detection.polygon.exterior.coords, np.int32)], True, (0, 255, 0), 1)

                    #draw the confidence
                    cv2.putText(drawImage, text, (int(detection.bbox[0]), int(detection.bbox[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                else: 
                    #find the closest point of the view to the detection polygon, when they do not intersect

                    viewSize = Detection.imageSize

                    centerX, centerY = viewSize[1]//2, viewSize[0]//2
                    radius = min(viewSize)//2
                    radius -= Detection.margin

                    detectionPoint = detection.polygon.centroid

                    closestPoint = closest_point_on_circle(centerX, centerY, radius, detectionPoint.x, detectionPoint.y)

                    cv2.circle(drawImage, (int(closestPoint[0]), int(closestPoint[1])), 3, (0, 0, 255), -1)
                    cv2.putText(drawImage, text, (closestPoint[0], closestPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
            trackedDetectionsImage = drawImage.copy()

            if doVideo:
                videoWriter.write(trackedDetectionsImage)
        if doDebug:
            for detection in newDetections:
                cv2.polylines(drawImage, [np.array(detection.polygon.exterior.coords, np.int32)], True, (0, 0, 255), 1)


            cv2.imshow("Tracked Detections", drawImage)




        



        


        return self.detectionsToPoints(self.trackedDetections)





def closest_point_on_circle(Cx, Cy, R, Px, Py):
    # Calculate vector from the center of the circle to the point
    vector_x = Px - Cx
    vector_y = Py - Cy
    
    # Calculate the distance from the center to the point
    distance = sqrt(vector_x**2 + vector_y**2)
    
    # Normalize the vector
    if distance != 0:
        unit_vector_x = vector_x / distance
        unit_vector_y = vector_y / distance
    else:
        # Point is at the center of the circle, return any point on the circle
        return (Cx + R, Cy)
    
    # Scale the vector by the radius to find the closest point on the circle
    Qx = Cx + R * unit_vector_x
    Qy = Cy + R * unit_vector_y
    
    return (int(Qx), int(Qy))



def main():
    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")

    #episodeManager.currentIndex = 
    episodeManager.nextEpisode(6)
    #episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    doVideo = True
    name = episode.name
    videoOutputPath = f"runs/{name}.mp4"
    fps = 10
    video_writer = None
    if doVideo:
        frame = episode[0].image
        height, width, layers = frame.shape
        #delete the video if it already exists
        
        if os.path.exists(videoOutputPath):
            os.remove(videoOutputPath)
        video_writer = cv2.VideoWriter(videoOutputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))





    branchModelTracker = BranchModelTracker("C:/Users/magnu/OneDrive/Misc/BronchoYolo/yolov5/runs/train/branchTraining8-XL/weights/best.pt")
    
    for index in range(len(episode)):
        

        if index < 150:
            continue

        frame = episode[index]

        #print(f"Processing frame {index}")
        #print(frame.image.shape)
        #findBranches(frame.image, doDraw=True)

        #watershed(frame.image)
        #watershed(frame.image)



        timeStamp1 = time.time()
        #contours = thresholdTree(frame.image, 1)
        timeStamp2 = time.time()

        #time taken in seconds
        #print(f"Time taken: {timeStamp2 - timeStamp1}")

        drawImage = frame.image.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            points, detections =branchModelTracker.predict(frame.image, doDebug=False, doVideo=doVideo, videoWriter=video_writer)


            for key, detection in detections.items():
                cv2.polylines(drawImage, [np.array(detection.polygon.exterior.coords, np.int32)], True, (0, 255, 0), 1)

                #draw the confidence
                cv2.putText(drawImage, f"{detection.confidence:.2f}", (int(detection.bbox[0]), int(detection.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #print bbox
                print(f"bbox {key}: {detection.bbox}")
        cv2.imshow("Threshold Contours", drawImage)

        #create_and_display_contour_tree(frame.image)
        
        key = cv2.waitKey(1)
        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
        # if esc is pressed, exit the loop
        elif key == 27:
            break


        
    
    if doVideo:
        video_writer.release()
        print(f"Video saved to {videoOutputPath}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





