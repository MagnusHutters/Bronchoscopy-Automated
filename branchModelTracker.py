
import copy
import numpy as np
import cv2
import torch
import time
import os
from shapely.geometry import Polygon, box, Point
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

    #cv2.imshow("processedImage", image)

    keypoints, descriptors = detector.detectAndCompute(image, None)









    return keypoints, descriptors


oldImage = None

def find_affine_transformation(img2, oldKeypoints, oldDescriptors, feature_type='AKAZE', featureScale=1.0):
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
    downsampleFactor = featureScale
    originalImage = img2
    global oldImage

    if doDownsample:
        img2 = cv2.resize(img2, (0, 0), fx=downsampleFactor, fy=downsampleFactor)

        
    


    # Detect features in both images
    #keypoints1, descriptors1 = detect_features(img1, feature_type)
    newKeypoints, newDescriptors = detect_features(img2, feature_type)

    if doDownsample:
        #upsample the keypoints to the original size

        for keypoint in newKeypoints:
            keypoint.pt = (keypoint.pt[0]/downsampleFactor, keypoint.pt[1]/downsampleFactor)

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
    if oldImage is not None:
        img_matches = cv2.drawMatches(oldImage, oldKeypoints, originalImage, newKeypoints, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv2.imshow("Matches", img_matches)
        #print(f"Number of matches: {len(matches)}")


    # Compute the affine transformation matrix
    affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if affine_matrix is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), newKeypoints, newDescriptors
    


    overallRotation = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])

    overallScaleX = sqrt(affine_matrix[0, 0] ** 2 + affine_matrix[1, 0] ** 2)

    overallScaleY = sqrt(affine_matrix[0, 1] ** 2 + affine_matrix[1, 1] ** 2)

    maxScale = 2.5
    minScale = 0.4
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



    oldImage = originalImage

    
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

    



    for yoloDetection in yolo_predictions:


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




def match_detections(list1, list2, min_similarity=0.2):
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


            lastMatched = float(det1.iterationsSinceLastMatch)
            lastMathcedThreshold = 15

            if lastMatched<=lastMathcedThreshold:  # Only apply the minimum similarity threshold if the detection has been matched recently

                cost_matrix[i, j] = -similarity if similarity >= min_similarity else 999999
            else:                       
                cost_matrix[i, j] = -similarity if similarity >= min_similarity* (lastMathcedThreshold/lastMatched) else 999999



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




class WrappedList:
        def __init__(self, list):
            self.list = list
            self.length = len(list)

        def __getitem__(self, index):
            if index >=0:
                return self.list[index % self.length]
            else:
                return self.list[index]
            
        def __setitem__(self, index, value):
            if index >=0:
                self.list[index % self.length] = value
            else:
                self.list[index] = value

        def __len__(self):
            return self.length
        
        def __str__(self):
            return str(self.list)
        
        def __repr__(self):
            return str(self.list)
        
        


class Detection:

    
    imageSize=(400,400)

    margin = 20
    
    viewPolygonWithMargin = Polygon([(margin, margin), (margin, imageSize[1]-margin), (imageSize[0]-margin, imageSize[1]-margin), (imageSize[0]-margin, margin)])

    viewPolygon = Polygon([(0, 0), (0, imageSize[1]), (imageSize[0], imageSize[1]), (imageSize[0], 0)])




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
        self.isMatched = False
        self.iterationsSinceLastMatch = 0
        
        self.id=id
        self.isNew = True
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
    def fromDict(cls, detection_dict):
        #initialize a detection object from a dictionary
        class_id = detection_dict["class_id"]
        confidence = detection_dict["confidence"]
        polygon = Polygon(detection_dict["polygon"])
        bbox = detection_dict["bbox"]
        id = detection_dict["id"]
        childrenIds = detection_dict["children"]
        parentId = detection_dict["parent"]
        inView = detection_dict["inView"]


        detection = cls(class_id, confidence, polygon, id, bbox)

        detection.inView = inView





        return detection, childrenIds, parentId

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
        



        
        

    




        #to bounds and back
        #self.polygon = box(*self.polygon.bounds)



        

        #create a list of the 4 corners of the polygon
        



                

        #self.polygon = self.calculatePartOfPolygonOutsideView()



        #self.polygonOutsideView = affine_transform(self.polygonOutsideView, affine_params)


        #self.polygonOutsideView = self.calculatePartOfPolygonOutsideView()

        #self.totalPolygon = self.polygon.union(self.polygonOutsideView)
        # Update bounding box after transformation
        self.bbox = self.polygon.bounds


    def calculateViewIntersection(self):
        self.intersectionWithView = self.polygon.intersection(self.viewPolygonWithMargin)

        self.inView = self.intersectionWithView.area > 1



    def squareify(self):
        '''
        Squareify the polygon if along the edge of the view






        '''



        #Squareify the polygon if along the edge of the view

        corners = list(self.polygon.exterior.coords[0:4])
        #print(f"corners: {corners}")

        #map corners to a list of points
        corners = WrappedList([Point(corner) for corner in corners])

        numpyCorners = WrappedList([np.array(corner.coords[0]) for corner in corners.list])


        def sideIsInsideOppositeIsOutside(corners, sideIndex):
        #check if the side is inside the view and the opposite side is outside
        
        #if the side is inside the view, return false
            return \
                self.viewPolygonWithMargin.contains((corners[sideIndex])) and \
                self.viewPolygonWithMargin.contains((corners[sideIndex+1])) and \
                not self.viewPolygonWithMargin.contains((corners[sideIndex+2])) and \
                not self.viewPolygonWithMargin.contains((corners[sideIndex-1]))
        
        def cornerIsInsideRestIsOutside(corners, cornerIndex):
            #check if the corner is inside the view and the rest is outside
            return \
                self.viewPolygonWithMargin.contains((corners[cornerIndex])) and \
                not self.viewPolygonWithMargin.contains((corners[cornerIndex-1])) and \
                not self.viewPolygonWithMargin.contains((corners[cornerIndex+1])) and \
                not self.viewPolygonWithMargin.contains((corners[cornerIndex+2]))


        foundSide = False
        for i in range(4):
            if sideIsInsideOppositeIsOutside(corners, i):
                
                insideLength = corners[i].distance(corners[i+1])

                outsideLength = corners[i+2].distance(corners[i+1])


                diff = insideLength - outsideLength
                maxDiff = insideLength * 0.1

                #clamp diff
                diff = max(0, min(maxDiff, diff))
                




                toOutsideVector = numpyCorners[i+2] - numpyCorners[i+1]
                normalizedToOutsideVector = toOutsideVector / outsideLength

                newOutsidePoint1 = numpyCorners[i-1] + normalizedToOutsideVector * diff
                newOutsidePoint2 = numpyCorners[i+2] + normalizedToOutsideVector * diff

                corners[i-1] = Point(newOutsidePoint1)
                corners[i+2] = Point(newOutsidePoint2)
                foundSide = True
                break

        if not foundSide:
            for i in range(4):
                if cornerIsInsideRestIsOutside(corners, i):

                    insideLength1 = corners[i].distance(corners[i-1])
                    insideLength2 = corners[i].distance(corners[i+1])

                    
                    dif1 = insideLength1 - insideLength2
                    dif2 = insideLength2 - insideLength1

                    maxDif = max(insideLength1, insideLength2) * 0.3
                    
                    dif1 = max(0, min(maxDif, dif1))
                    dif2 = max(0, min(maxDif, dif2))


                    toOutsideVector1 = numpyCorners[i-1] - numpyCorners[i]
                    toOutsideVector2 = numpyCorners[i+1] - numpyCorners[i]

                    normalizedToOutsideVector1 = toOutsideVector1 / insideLength1
                    normalizedToOutsideVector2 = toOutsideVector2 / insideLength2

                    newOutsidePoint1 = numpyCorners[i-1] + normalizedToOutsideVector1 * dif2
                    newOutsidePoint2 = numpyCorners[i+1] + normalizedToOutsideVector2 * dif1

                    newOutsidePoint3 = numpyCorners[i+2] + normalizedToOutsideVector1 * dif2 + normalizedToOutsideVector2 * dif1

                    corners[i-1] = Point(newOutsidePoint1)
                    corners[i+1] = Point(newOutsidePoint2)
                    corners[i+2] = Point(newOutsidePoint3)

                    break

        self.polygon = Polygon([corner.coords[0] for corner in corners.list])

        self.bbox = self.polygon.bounds

        self.calculateViewIntersection()   

                

        #extra decay depending on size compared to view
        


    def apply_match(self, matched_detection, interpolation_factor=0.8, currentKey=-1):
        """
        Applies the result of a match to this detection, updating the polygon and confidence.

        Parameters:
        matched_detection (Detection or None): The matched detection object, or None if no match.
        """
        def interpolate_box(box1, box2, t=0.5):
            """Interpolate two bounding boxes"""
            minx = box1[0] * (1 - t) + box2[0] * t
            miny = box1[1] * (1 - t) + box2[1] * t
            maxx = box1[2] * (1 - t) + box2[2] * t
            maxy = box1[3] * (1 - t) + box2[3] * t
            return (minx, miny, maxx, maxy)

        def apply_interpolation(orignial_box, target_box, alt_box, detection_box, mask_poly):
            
            """Directly adjust vertices to new positions from the interpolated box if inside the mask polygon."""
            new_coords = []
            

            maskMinX, maskMinY, maskMaxX, maskMaxY = mask_poly.bounds


            origMinX, origMinY, origMaxX, origMaxY = orignial_box
            targetMinX, targetMinY, targetMaxX, targetMaxY = target_box
            altMinX, altMinY, altMaxX, altMaxY = alt_box
            detMinX, detMinY, detMaxX, detMaxY = detection_box

            if maskMinX < origMinX < maskMaxX:
                newMinX = targetMinX
            else:

                #if both detection x values are inside the mask, use the target value
                if maskMinX < detMinX < maskMaxX and maskMinX < detMaxX < maskMaxX:
                    #print("MinX - Both inside")
                    newMinX = altMinX
                else:
                    newMinX = origMinX

                

            if maskMinY < origMinY < maskMaxY:
                newMinY = targetMinY
            else:
                
                #if both detection y values are inside the mask, use the target value
                if maskMinY < detMinY < maskMaxY and maskMinY < detMaxY < maskMaxY:

                    #print("MinY - Both inside")
                    newMinY = altMinY
                else:
                    newMinY = origMinY

            if maskMinX < origMaxX < maskMaxX:
                newMaxX = targetMaxX
            else:
                
                #if both detection x values are inside the mask, use the target value
                if maskMinX < detMinX < maskMaxX and maskMinX < detMaxX < maskMaxX:
                    #print("MaxX - Both inside")
                    newMaxX = altMaxX
                else:
                    newMaxX = origMaxX

            if maskMinY < origMaxY < maskMaxY:
                newMaxY = targetMaxY
            else:
                
                #if both detection y values are inside the mask, use the target value
                if maskMinY < detMinY < maskMaxY and maskMinY < detMaxY < maskMaxY:
                    #print("MaxY - Both inside")
                    newMaxY = altMaxY
                else:
                    newMaxY = origMaxY

            #create polygon from bounds
            new_coords = [(newMinX, newMinY), (newMinX, newMaxY), (newMaxX, newMaxY), (newMaxX, newMinY)]

            
            return Polygon(new_coords)



        self.calculateViewIntersection()

        if matched_detection is not None:
            self.isNew = False
            if self.iterationsSinceLastMatch <= 2:
                self.iterationsSinceLastMatch = 0
            self.iterationsSinceLastMatch = self.iterationsSinceLastMatch //2
            # Update polygon (for simplicity, we'll just keep the current polygon, but this could be extended)
            # Update confidence by adding the matched detection's confidence
            self.confidence += matched_detection.confidence


            unionPolygon = self.polygon.union(matched_detection.polygon)

            detectionBox = matched_detection.polygon.bounds
            unionBbox = unionPolygon.bounds

            interpolatedBbox = interpolate_box(unionPolygon.bounds, matched_detection.bbox, interpolation_factor)
            interpolatedBbox1 = interpolate_box(unionPolygon.bounds, matched_detection.bbox, interpolation_factor*0.3)

            self.polygon = apply_interpolation(unionBbox, interpolatedBbox, interpolatedBbox1, detectionBox, self.viewPolygon)

            self.bbox = self.polygon.bounds


            #if polygon has more than 10 points, simplify it
            #if len(self.polygon.exterior.coords) > 4:
            #    self.polygon = self.polygon.simplify(1)

            self.bbox = self.polygon.bounds

            if self.id != currentKey:
                self.confidence *= 0.99






        else: # No match - handle this case


            self.iterationsSinceLastMatch += 1
            # Decrease confidence by a fixed amount or percentage if no match

            #if not matched squareify the polygon



            children = self.children
            anyChildrenHasMatch= False

            if self.id != currentKey:
                for child in children:
                    if child.isMatched:
                        anyChildrenHasMatch = True
                        #print("Child has match")
                        
                        #found alive child - decrease confidence greatly
                        self.confidence *= 0.8

            
                
            #if new and no match, decrease confidence
            if self.isNew:
                self.confidence *= 0.75


            
            viewArea = self.viewPolygon.area
            detectionArea = self.polygon.area

            detectionToViewRatio = detectionArea/viewArea


            detectionHeight = self.polygon.bounds[3] - self.polygon.bounds[1]
            detectionWidth = self.polygon.bounds[2] - self.polygon.bounds[0]

            maxDetectionSide = max(detectionHeight, detectionWidth)
            maxDetectionSideRation = maxDetectionSide / 400.0

            rectangleRatio = detectionHeight/detectionWidth
            if rectangleRatio < 1:
                rectangleRatio = 1/rectangleRatio

            

            if self.inView:
            
                self.confidence *= 0.98

                if self.confidence < 0.1:
                    #remove the detection

                    self.confidence = 0

                

                #rectangleRatioLimit = 2 #if more than twice as high as wide, or opposite, decrease confidence
                #if rectangleRatio > rectangleRatioLimit:
                #    decay = 0.02 * (rectangleRatio-rectangleRatioLimit)
                #    self.confidence *= 1-decay
                if self.id != currentKey:
                    if detectionToViewRatio > 0.25:
                        decay = 0.01 * detectionToViewRatio
                        self.confidence *= 1-decay

                    if detectionToViewRatio > 0.75:
                        
                        self.confidence *= 0.8

                    if detectionToViewRatio < 0.001:
                        self.confidence *= 0.8

                    if maxDetectionSideRation > 0.5:
                        decay = 0.005 * maxDetectionSideRation
                        self.confidence *= 1-decay


            else:

                if self.id != currentKey:
                    self.confidence *= 0.98

                    distance = self.polygon.centroid.distance(self.viewPolygon.centroid)

                    if distance > 1200:
                        self.confidence *= 0.9

                    if detectionToViewRatio > 1.5:
                        
                        self.confidence *= 0.9
                


                    if self.confidence < 0.1:
                        #remove the detection

                        self.confidence = 0




        
        


            


                                                             

    def isInsided(self, detection, insideCriteria=0.7): #check if the other detection is inside this detection, counts as inside if the intersection is more than 80% of this detection and not the opposite
        intersection = self.polygon.intersection(detection.polygon)

        #intersection is more than % of this detection         - at least % of this detection is inside other detection
        isInsideOther = intersection.area > insideCriteria * self.polygon.area              
        #intersection is less than % of the other detection    - no more than % of the other detection is inside this detection
        otherNotInsideThis = intersection.area < insideCriteria * detection.polygon.area    


        return isInsideOther and otherNotInsideThis

    def contains(self, detection, insideCriteria=0.7): #check if this detection contains the other detection, counts as inside if the intersection is more than 80% of the other detection

        return detection.isInsided(self, insideCriteria)      

    def get_intersection(self, other_detection):
        """
        Computes the intersection area with another detection.

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The intersection area.
        """
        

        return self.polygon.intersection(other_detection.polygon)

    def get_union(self, other_detection):
        """
        Computes the union area with another detection.

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The union area.
        """
        return self.polygon.union(other_detection.polygon)

    def get_similarity(self, other_detection):
        """
        Computes the similarity score with another detection using Intersection over Union (IoU).

        Parameters:
        other_detection (Detection): Another detection object.

        Returns:
        float: The similarity score (IoU).
        """


        
        

        intersection = self.get_intersection(other_detection)
        union = self.get_union(other_detection)

        #only use part in view
        intersection = intersection.intersection(self.viewPolygon)
        union = union.intersection(self.viewPolygon)

        intersection_area = intersection.area
        union_area = union.area


        #if self has children, check that all children are inside the other detection
        if len(self.children) > 0:
            for child in self.children:
                #check if child is in view
                if child.inView:
                    
                    #if intersection is more than 0
                    intersection = child.polygon.intersection(other_detection.polygon).area
                    if intersection == 0:    
                    
                        #("Child not inside")
                        return 0




        return intersection_area / union_area if union_area != 0 else 0

class BranchModelTracker:
    def __init__(self, modelPath, featureScale=1.0):
        pass

        #load the model
        self.featureScale = featureScale

        self.model = torch.hub.load('BronchoYolo/yolov5', 'custom', path=modelPath, source='local', force_reload=True)


        self.model = self.model.to('cuda')
        self.model.eval()

        torch.cuda.empty_cache()

        print(f"Parameters: {next(self.model.parameters()).device}")



        self.oldImage = None
        self.newId = 0


        self.trackedDetections = None

        self.oldSescriptors = None
        self.oldKeypoints = None
    

    def reset(self):
        self.trackedDetections = None
        self.oldSescriptors = None
        self.oldKeypoints = None
        self.newId = 0
        self.oldImage = None

    def detectionsToPoints(self, detections, currentKey = -1):
        #takes the center of the bounding box of the detections and returns them as a list of points

        buildHierachy(detections)



        finalDetections={}

        #certaintyCutoff = 0.1
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

                #if has grandchild, don't add
                for child in detection.children:
                    if len(child.children) > 0:
                        doAdd = False

                #if detection is the current key, add it - always show the currently selected detection, if it exists
                if detection.id == currentKey:
                    doAdd=True

                

                #doAdd=True

                
                if doAdd:
                    if detection.inView:
                            
                        point = detection.polygon.centroid
                        points[detection.id] = (point.x, point.y)
                    else:
                        viewSize = Detection.imageSize

                        centerX, centerY = viewSize[1]//2, viewSize[0]//2
                        radius = min(viewSize)//2
                        radius -= Detection.margin

                        closestPoint = detection.polygon.centroid

                        #closestPoint = closest_point_on_circle(centerX, centerY, radius, closestPoint.x, closestPoint.y)

                        points[detection.id] = (int(closestPoint.x), int(closestPoint.y))

                    finalDetections[detection.id] = copy.deepcopy(detection)
                


        return points, finalDetections





    def getPredictions(self, image):

        #disable warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
        
            results = self.model(image)

        return results

    def apply_matches(self,matches, trackedDetections, currentKey = -1):
        """
        Applies matches to a list of detections.

        Parameters:
        matches (list): A list of tuples, where each tuple contains a detection from list1 and its matched detection
                        from list2 or None if no match was found.
        """
        #print(f"matches: {matches}")

        for det1, det2 in matches:

            if det1 is not None:
                if det2 is not None:
                    det1.isMatched = True
                else:
                    det1.isMatched = False


        for det1, det2 in matches:
            if det1 is not None:
                det1.apply_match(det2, currentKey= currentKey)

                det1.squareify()
            else:
                #add the unmatched new detection to the list of tracked detections
                det2.id = self.newId
                self.newId += 1
                det2.squareify()

                trackedDetections.append(det2)

                
        trackedDetections = [detection for detection in trackedDetections if detection.confidence > 0]


        return trackedDetections

    def predict(self, image, doDebug=False, doVideo=False, videoWriter=None, active = True, currentKey = -1):


        def filter_corner_detections(detections, image_width, image_height, corner_margin=5, area_threshold=0.02):
            """
            Filters out small detections near the corresponding corners of an image.
            
            :param detections: List of detections, each detection is a dictionary with 'x1', 'y1', 'x2', 'y2' representing the bounding box.
            :param image_width: Width of the image.
            :param image_height: Height of the image.
            :param corner_margin: Margin in pixels to consider as 'corner'. Default is 20 pixels.
            :param area_threshold: Fraction of image area that a detection must exceed to be kept. Default is 0.02 (2%).
            
            :return: Filtered list of detections.
            """
            
            filtered_detections = []
            image_area = image_width * image_height
            min_area = image_area * area_threshold  # Calculate the minimum area threshold

            def is_bbox_near_image_corner(x1, y1, x2, y2):
                """
                Check if the top-left, top-right, bottom-left, or bottom-right corner of the bounding box is near 
                the corresponding corner of the image.
                """
                # Top-left corner check
                if x1 <= corner_margin and y1 <= corner_margin:
                    return True
                # Top-right corner check
                if x2 >= image_width - corner_margin and y1 <= corner_margin:
                    return True
                # Bottom-left corner check
                if x1 <= corner_margin and y2 >= image_height - corner_margin:
                    return True
                # Bottom-right corner check
                if x2 >= image_width - corner_margin and y2 >= image_height - corner_margin:
                    return True

                return False

            for detection in detections:
                x1, y1, x2, y2 = bbox = detection[:4]
                detection_area = (x2 - x1) * (y2 - y1)
                
                # If the detection is not in the corner or has an area larger than the threshold, keep it
                if not is_bbox_near_image_corner(x1, y1, x2, y2) or detection_area >= min_area:
                    filtered_detections.append(detection)
            
            return filtered_detections


        if not active:
            return {}, {}

        report=Timer.reset()
        #print(report)
        Timer.point("predict")



        newImage = image
        oldImage = self.oldImage

        self.oldImage = newImage

        

        Timer.point("getPredictions")


        time1 = time.time()
        newPredictions = self.getPredictions(newImage)

        Timer.point("Got Yolo Predictions")
        newPredictions = newPredictions.pred[0]

        newPredictions = filter_corner_detections(newPredictions, newImage.shape[1], newImage.shape[0], corner_margin=2, area_threshold=0.05)

        time2 = time.time()
        #print(f"YOLO inference time: {time2-time1} seconds")
        

        

        #print(newPredictions)

        Timer.point("createDetections")
        newDetections = create_detections_from_yolo(newPredictions)

        
        #squareify the detections
        #for detection in newDetections:
        #    detection.squareify()

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

            buildHierachy(self.trackedDetections)
            return self.detectionsToPoints(self.trackedDetections)


        Timer.point("findAffineTransformation")

        time3 = time.time()

        
        #affineTransformation = find_affine_transformation(oldImage, newImage, feature_type='AKAZE')
        affineTransformation, self.oldKeypoints, self.oldSescriptors = find_affine_transformation(newImage, self.oldKeypoints, self.oldSescriptors, feature_type='AKAZE', featureScale = self.featureScale)


        
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
        self.trackedDetections=self.apply_matches(matchedPairs,self.trackedDetections, currentKey = currentKey)


        
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




        



        buildHierachy(self.trackedDetections)


        return self.detectionsToPoints(self.trackedDetections, currentKey = currentKey)





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
    episodeManager = EpisodeManager(mode = "Read", loadLocation="DatabaseLabelled/")

    #episodeManager.currentIndex = 
    #shuffle the episodes
    #episodeManager.doShuffleEpisodes()
    doExit = False

    while not doExit:
        episodeManager.nextEpisode()
        #episodeManager.nextEpisode()

        episode = episodeManager.getCurrentEpisode()

        doVideo = False
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
            

            if index < 50:
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
                points, detections =branchModelTracker.predict(frame.image, doDebug=True, doVideo=doVideo, videoWriter=video_writer)


                for key, detection in detections.items():
                    cv2.polylines(drawImage, [np.array(detection.polygon.exterior.coords, np.int32)], True, (0, 255, 0), 1)

                    #draw the confidence
                    cv2.putText(drawImage, f"{detection.confidence:.2f}", (int(detection.bbox[0]), int(detection.bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    #print bbox
                    #(f"bbox {key}: {detection.bbox}")
            cv2.imshow("Threshold Contours", drawImage)

            #create_and_display_contour_tree(frame.image)
            
            key = cv2.waitKey(1)
            # If the user presses 'q', exit the loop
            if key == ord('q'):
                doExit = True
                break
            # if esc is pressed, exit the loop
            elif key == 27:
                doExit = True
                break

        
    
        if doVideo:
            video_writer.release()
            print(f"Video saved to {videoOutputPath}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





