

from DataHandling.Episode import EpisodeManager, Episode

import numpy as np
import cv2
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
import time

import matplotlib.pyplot as plt

from collections import defaultdict

#chapely for polygon intersection
from shapely.geometry import Polygon

mask = None


def createVignetteMask(image):
    rows, cols = image.shape


    radius_ratio=1.15
    strength = 1 # bigger makes the vignette softer, smaller makes it harder

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


def preProcessImage(image, downscaleFactor=1):
    global mask

    #print(f"Image: {image}")

    #downscale the image
    newShape = (image.shape[1] // downscaleFactor, image.shape[0] // downscaleFactor)
    image = cv2.resize(image, newShape)

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

    


    #cv2.imshow("Mask", mask)

    # Apply the mask to the image
    vignette = image * mask
    

    # Normalize the image
    vignette = cv2.normalize(vignette, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    vignette = cv2.bitwise_not(vignette)

    return vignette






class Point:
    def __init__(self, coords, contour, depth, area, circumference, roundness):
        self.coords = coords
        self.contour = contour
        self.depth = depth
        self.area = area
        self.circumference = circumference
        self.roundness = roundness
        self.children = []

        self.hasChildren = False
        self.hasParent = False
        self.hasGrandChildren = False
        
        self.subNodes = []
        self.subNodes.append(self)


        #self.contours.append(contour)
        self.size=1
        self.parent = None

    def addChild(self, child):
        self.children.append(child)


    def removeChild(self, child):
        if child in self.children:
            self.size += child.size
            self.children.remove(child)


        

    def isInside(self, possilbeParent):
        for point in self.contour:
            #print(f"Point in self.contour: {point}, type: {type(point)}, shape: {point.shape}")

            # Convert point to tuple of two integers
            #point_tuple = tuple(point[0]) if point.ndim > 1 else tuple(point)
            point_tuple = (int(point[0][0]), int(point[0][1]))

            #print(f"Testing point: {point_tuple}, type: {type(point_tuple)}")
            #if len(point_tuple) != 2 or not all(isinstance(i, int) for i in point_tuple):
            #    raise ValueError(f"Point {point_tuple} is not a tuple of two integers")

            dist = cv2.pointPolygonTest(possilbeParent.contour, point_tuple, False)
            if dist < 0:
                return False
        return True

        #check if contour is inside the parent contour


    def setParent(self, parent):
        self.parent = parent
        self.parent.addChild(self)

    

    def concatenate(self):
        #if it has only one child, merge with the child
        if len(self.children) == 1:
            newCombinedSize = self.size + self.children[0].size
            
            if newCombinedSize > 64:
                return False





            child = self.children[0]
            
            if self.parent is not None:
                child.parent = self.parent
                child.size += self.size
                
                

                child.subnodes = self.subNodes + child.subNodes


                child.contour = self.contour
                child.area = self.area
                child.circumference = self.circumference
                child.roundness = self.roundness

                self.parent.removeChild(self)
                self.parent.addChild(child)
                
                self.parent = None

                return True
        return False

    def trim(self, maxSize=2):
        

        #remove self if it has no children
        if len(self.children) == 0 and self.size <=maxSize:
            if self.parent is not None:
                self.parent.size += self.size
                self.parent.removeChild(self)

                
            return True
        
        
        return False


def drawNode(image, point, range, depth):
    
    



    numChildren = len(point.children)

    #divide the range into numChildren parts
        

    for i, child in enumerate(point.children):

        currentRangeSize = range[1] - range[0]
        childRangeSize = currentRangeSize / numChildren

        childRangeStart = int(range[0] + i * childRangeSize)
        childRangeEnd = int(range[0] + (i + 1) * childRangeSize)
        

        newRange = [childRangeStart, childRangeEnd]
        newDepth = depth + 1


        #draw the line from the center of the 2 ranges, using the depth as x coordinate
        x1 = depth*2
        x2 = newDepth*2
        y1 = int((range[0] + range[1]) / 2)
        y2 = int((childRangeStart + childRangeEnd) / 2)

        cv2.line(image, (y1, x1), (y2, x2), (255, 255, 255), 1)

        drawNode(image, child, newRange, newDepth)
        

        





def findBranches(originalImage, contourDepth=0.5, doDraw=False):
    

    image = preProcessImage(originalImage)
    
    #display the image
    if doDraw:
        cv2.imshow("Preprocessed Image", image)

    points = []





    layers = []
    allPoints = []
    
    lastLayer = []
    for threshold in range(255, 0, -1):


        layer=[]

        depth = 255-threshold
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

            if area < 4:
                continue


            roundness = 0
            if circumference == 0:
                roundness = 0
            else:
                roundness = 4 * np.pi * area / (circumference * circumference)

            center= (0,0)


            

            

            #calculate the center of mass
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

            newPoint = Point(center, contour, depth, area, circumference, roundness)

            hasParent = False
            for lastLayerPoint in lastLayer:
                if newPoint.isInside(lastLayerPoint):
                    newPoint.setParent(lastLayerPoint)
                    hasParent = True
                    break
                #no parent found
                
            #if no parent found, widden the search, and find the closest parent
            if not hasParent:
                minDistance = 999999999
                closestParent = None
                for lastLayerPoint in lastLayer:
                    distance = cv2.pointPolygonTest(lastLayerPoint.contour, center, True)
                    if distance < minDistance:
                        minDistance = distance
                        closestParent = lastLayerPoint

                if closestParent is not None:
                    newPoint.setParent(closestParent)
                    hasParent = True
            if hasParent or depth <=1:

                layer.append(newPoint)
                allPoints.append(newPoint)
            else:
                print("No parent found")

        layers.append(layer)
        lastLayer = layer



    lines = [] # List of lines to draw between 3d points

    for point in allPoints: # For each point, aheck if it has a parent, and add a line between them, using depth as z coordinate
        if point.parent is not None:
            lines.append([point.coords[0], point.coords[1], point.depth, point.parent.coords[0], point.parent.coords[1], point.parent.depth])

    
    #Trim the tree
    if doDraw:
        drawImage = originalImage.copy()
        #drawImagePre = originalImage.copy()
        drawImageOrig = originalImage.copy()


        

        for point in allPoints:
                
            #draw line to parent
            if point.parent is not None:
                cv2.line(drawImageOrig, point.coords, point.parent.coords, (0, 255-point.depth, point.depth), 1)

        cv2.imshow("ImageOrig", drawImageOrig)



    #cv2.imshow("ImagePre", drawImagePre)
    trim=7
    iterations = 4
    for i in range(iterations):
        progress = (i+1) / iterations
        toTrim = int(trim*progress)

        newAllPoints = []
        for point in allPoints:
            hasBeenTrimmed = point.trim(trim)
            if not hasBeenTrimmed:
                newAllPoints.append(point)
        allPoints = newAllPoints


        newAllPoints = []
        for point in allPoints:
            hasBeenTrimmed = point.concatenate()
            if not hasBeenTrimmed:
                newAllPoints.append(point)
        allPoints = newAllPoints

        
    

    for point in allPoints:
        
        #draw line to parent
        if point.parent is not None:
            point.hasParent = True

            hasGrandChildren = False
            for child in point.children:
                point.hasChildren = True
                if len(child.children) > 0:
                    point.hasGrandChildren = True
                    hasGrandChildren = True
                    break
            


    candidates = []
    for point in allPoints:
        if not point.hasGrandChildren and point.hasParent:
            


            #Contruct the candidate from relevant subnode
            amountSubNodes = len(point.subNodes)
            relevantSubNode = int(amountSubNodes*contourDepth)
            subnode=point.subNodes[relevantSubNode]

            contour = point.contour
            area = point.area
            circumference = point.circumference
            roundness = point.roundness
            center = point.coords
            
            if area < 4:
                continue

            if area > 0.5 * image.shape[0] * image.shape[1]:
                continue

            candidate = Point(center, contour, point.depth, area, circumference, roundness)

            candidate.size = point.size
            candidate.hasChildren = point.hasChildren
            candidate.hasParent = point.hasParent
            candidate.hasGrandChildren = point.hasGrandChildren


            candidates.append(candidate)

    
    
    #draw the centers

    if doDraw:
        for point in allPoints:
            
            #draw line to parent
            if point.parent is not None:


                #


                #find relevant subnode
                amountSubNodes = len(point.subNodes)
                relevantSubNode = int(amountSubNodes*0.5)
                subnode=point.subNodes[relevantSubNode]

                #find relevant parent subnode
                amountSubNodes = len(point.parent.subNodes)
                relevantSubNode = int(amountSubNodes*0.5)
                parentSubnode=point.parent.subNodes[relevantSubNode]


                cv2.line(drawImage, subnode.coords, parentSubnode.coords, (0, 255-point.depth, point.depth), 1)


                #if area is smaller than percentage of image




                #if any of the children has children, skip
                hasGrandChildren = False
                for child in point.children:
                    if len(child.children) > 0:
                        hasGrandChildren = True
                        break
                if not hasGrandChildren:

                    


                    #generate unique random color 
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    #normalize the color
                    magnitude = np.sqrt(color[0]**2 + color[1]**2 + color[2]**2)
                    color = (color[0]/magnitude*255, color[1]/magnitude*255, color[2]/magnitude*255)
                    #to int
                    color = (int(color[0]), int(color[1]), int(color[2]))



                    cv2.circle(drawImage, subnode.coords, 8, color, -1)


                    





                    cv2.drawContours(drawImage, [subnode.contour], 0, color, 1)

                    boundingBox = cv2.boundingRect(subnode.contour)
                    cv2.rectangle(drawImage, (boundingBox[0], boundingBox[1]), (boundingBox[0]+boundingBox[2], boundingBox[1]+boundingBox[3]), color, 1)
                
                    
        
        cv2.imshow("Image", drawImage)

    return candidates


#================================================================================================
#================================= Shape Matching Functions =====================================
#================================================================================================

def average_distance_with_pointPolygonTest(contour1, contour2):
    total_distance1_to_2 = 0
    for point in contour1:

        coord = (int(point[0][0]), int(point[0][1]))
        distance = cv2.pointPolygonTest(contour2, coord, True)
        total_distance1_to_2 += abs(distance)
    average_distance1_to_2 = total_distance1_to_2 / len(contour1)
    
    total_distance2_to_1 = 0
    for point in contour2:
        coord = (int(point[0][0]), int(point[0][1]))
        distance = cv2.pointPolygonTest(contour1, coord, True)
        total_distance2_to_1 += abs(distance)
    average_distance2_to_1 = total_distance2_to_1 / len(contour2)
    
    return (average_distance1_to_2 + average_distance2_to_1) / 2

def size_similarity(contour1, contour2):
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    if area1 == 0 or area2 == 0:
        return float('inf')  # Avoid division by zero
    return max(area1, area2) / min(area1, area2)




# Function to calculate the combined similarity score between two contours, based on shape, position, and size - smaller is better


def combined_similarity(contour1, contour2, weight_shape=1.0, weight_position=1.0, weight_size=1.0):
    shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    avg_distance = average_distance_with_pointPolygonTest(contour1, contour2)
    size_similarity_score = size_similarity(contour1, contour2)
    combined_similarity_score = (weight_shape * shape_similarity + weight_position * avg_distance + weight_size * size_similarity_score) / (weight_shape + weight_position + weight_size)
    return combined_similarity_score



#match contours based on similarity using the hungarian algorithm
def matchBranches(contours1, contours2, threshold=5000, weight_shape=5.0, weight_position=10.0, weight_size=1.0):

    # Create a cost matrix with the similarity scores between each pair of contours
    num_contours1 = len(contours1)
    num_contours2 = len(contours2)
    cost_matrix = np.zeros((num_contours1, num_contours2))
    for i, contour1 in enumerate(contours1):
        for j, contour2 in enumerate(contours2):
            cost_matrix[i, j] = combined_similarity(contour1, contour2, weight_shape, weight_position, weight_size)


    # Use the Hungarian algorithm to find the optimal matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] < threshold:
            matches.append((i, j))

    return matches



    
def watershed(image):

    #resize the image
    image = cv2.resize(image, (100, 100))

    displayImage = image.copy()
    waterfalImage = image.copy()


    #preprocess the image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    #image = cv2.GaussianBlur(image, (3, 3), 0)
    # Peform closing operation to remove small holes
    kernel = np.ones((3,3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    #blur the image
    image = cv2.GaussianBlur(image, (5,5), 0)

    # Perform opening operation to remove noise
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

    #display the image
    cv2.imshow("WaterShed Preprqocessed Image", image)


    #find local minima


    #dynamic thresholding to find sure foreground
    ret, sure_fg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    threshold = 127
    #sure_fg = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]

    #display the image
    cv2.imshow("WaterShed Sure FG", sure_fg)



    local_min = peak_local_max(-image, min_distance=4, threshold_abs=1, labels=sure_fg)

    #display local minima

    for point in local_min:
        cv2.circle(displayImage, (point[1], point[0]), 3, (255, 255, 255), -1)
    cv2.imshow("WaterShed Local Min", displayImage)
    
    

    regionIntensities = {}
    #create markers
    markers = np.zeros(image.shape, dtype=int)
    for i, point in enumerate(local_min):

        region = i+1
        intensity = image[point[0], point[1]]
        regionIntensities[region] = intensity

        markers[point[0], point[1]] = region




    #gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    watershed = cv2.watershed(waterfalImage, markers)




    boundaryIndices = np.where(watershed == -1)

    #defualt dict with float("inf")
    boundaryIntensities = defaultdict(lambda: float("inf"))
    boundaryRegions = defaultdict(list)
    boundaryConers = [] #list of tuples with the coordinates of the boundary corners where more than 2 regions meet, need to be rechecked if regions are merged

    
    #transpose
    for index, point in enumerate(zip(boundaryIndices[0], boundaryIndices[1])):
        x = point[0]
        y = point[1]

        xmin = max(0, x-1)
        xmax = min(image.shape[0], x+1)
        ymin = max(0, y-1)
        ymax = min(image.shape[1], y+1)

        window = watershed[xmin:xmax, ymin:ymax]

        uniqueLabels = np.unique(window)
        #remove border and background¨
        uniqueLabels = uniqueLabels[uniqueLabels > 0]

        if len(uniqueLabels) ==2:
            region1 = min(uniqueLabels[0], uniqueLabels[1])
            region2 = max(uniqueLabels[0], uniqueLabels[1])
            boundaryRegions[(region1, region2)].append((x, y))
            intensity = image[x, y]
            if intensity < boundaryIntensities[(region1, region2)]:
                boundaryIntensities[(region1, region2)] = intensity

        if len(uniqueLabels) > 2:
            boundaryConers.append((x, y))

    

    for regionPair, intensity in boundaryIntensities.items():
        region1 = regionPair[0]
        region2 = regionPair[1]
        
        region1Intensity = regionIntensities[region1]
        region2Intensity = regionIntensities[region2]

        borderIntensity = intensity

        dif1 = abs(region1Intensity - borderIntensity)
        dif2 = abs(region2Intensity - borderIntensity)

        minDif = min(dif1, dif2)




        






    figure = plt.figure()
    plt.imshow(watershed, cmap='nipy_spectral')

    plt.show()


    #display the image
    #cv2.imshow("WaterShed Local Min", local_min)

    #display the image

    #



def contourOverlaps(contour1, contour2):

    for point in contour2:
        coord = (int(point[0][0]), int(point[0][1]))
        distance = cv2.pointPolygonTest(contour1, coord, True)
        if distance > 0:
            return True
    return False



class Contour:
    def __init__(self, contour, initialSize=1, depth=0, index=0):
        self.contour = contour
        self.size = initialSize
        self.children = []
        self.parent = None

        #calculate the center of contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.center = (cX, cY)
        else:
            self.center = (0, 0)
        self.lowestPoint=self.center
        
        self.depth = depth
        self.index = index
        


    def getStats(self):
        #calculate the area, circumference, and roundness of the contour
        area = cv2.contourArea(self.contour)
        circumference = cv2.arcLength(self.contour, True)
        roundness = 0
        if circumference == 0:
            roundness = 0
        else:
            roundness = 4 * np.pi * area / (circumference * circumference)

        childrenIndices = [child.index for child in self.children]
        siblingIndices = [sibling.index for sibling in self.parent.children if sibling != self] if self.parent is not None else []
        parentIndex = self.parent.index if self.parent is not None else -1
        numChildren = len(self.children)
        numSiblings = self.numSiblings()
        return self.contour, area, circumference, roundness, self.center, self.lowestPoint, self.size, self.depth,self.index, numChildren, numSiblings, childrenIndices, siblingIndices, parentIndex
    
    

    def totalBranchSize(self):
        size = self.size
        childSize=0
        for child in self.children:
            childSize = max(childSize, child.totalBranchSize())
        return size+childSize

    def contains(self, contour):

        for point in contour.contour:
            coord = (int(point[0][0]), int(point[0][1]))
            distance = cv2.pointPolygonTest(self.contour, coord, True)
            if distance >= 0:
                return True
        return False
    
    def consume(self, contour):
        self.size += contour.size
        self.children += contour.children

        for child in contour.children:
            child.parent = self

        self.lowestPoint = contour.lowestPoint
        self.depth = contour.depth

    def addChild(self, contour):
        self.children.append(contour)
        contour.parent = self


    

    def addChildren(self, contours):
        for contour in contours:
            self.addChild(contour)

    def hasChildren(self):
        return len(self.children) > 0
    def hasParent(self):
        return self.parent is not None
    def hasGrandChildren(self):
        for child in self.children:
            if child.hasChildren():
                return True
        return False
    def hasGrandParent(self):
        if self.parent is not None:
            return self.parent.hasParent()
        return False
    
    def numSiblings(self):
        if self.parent is not None:
            return len(self.parent.children) - 1

        return 0

    @staticmethod
    def contours(contours, initialSize=1):
        return [Contour(contour, initialSize, index=index) for index, contour in enumerate(contours)]

def thresholdTree(image, downscaleFactor=1):


    minimumAreaSize = 4
    minimumAreaSize = minimumAreaSize // downscaleFactor
    minimumArea=minimumAreaSize*minimumAreaSize
    image = preProcessImage(image, downscaleFactor)


    minimumIntensity = int(np.min(image))

    maximumIntensitySearchThreshold = 100

    finalContours = []

    activeContours = []

    stepSize = 2

    pruneSize = 4

    maxSize = 32

    
    for threshold in range(minimumIntensity, maximumIntensitySearchThreshold, stepSize):
        #inverse binary threshold
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

        #find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = Contour.contours(contours, initialSize=stepSize)

        #prune contours below a certain size
        contours = [contour for contour in contours if cv2.contourArea(contour.contour) > minimumArea]


        #print(f"")
        #print(f"=====================================================================================")

        #print(f"=====================================================================================")
        #print(f"Threshold: {threshold}, Number of contours: {len(contours)}")

        for contour in contours:

            #check if contour contains any active contours
            #if it does, check how many, if multiple prune any below a certain size, except the largest one
            #if there is only one, consume it
            #if it contains multiple move contained contours to finalContours


            #check if contour contains any active contours
            #print("")
            #print(f"=====================================================================================")
            #print(f"Checking contour [{contour.index}]")
            containedContours = []
            for i in range(len(activeContours)-1, -1, -1):
                activeContour = activeContours[i]

                if contour.contains(activeContour):
                    containedContours.append(activeContour)
                    activeContours.pop(i)
                    #print(f"Contour [{contour.index}] contains [{activeContour.index}]")

            #print(f"Number of contained contours: {len(containedContours)}")
            if len(containedContours) > 1:
                #multiple contained contours, prune any below a certain size, except the largest one
                containedContours.sort(key=lambda x: x.totalBranchSize(), reverse=True)

                #print(f"Contained contours have sizes: {[contour.size for contour in containedContours]}")
                
                #beforePruneSize = len(containedContours)
                containedContours = [containedContours[0]] + [contour for contour in containedContours[1:] if contour.size > pruneSize]
                #afterPruneSize = len(containedContours)


                #print(f"Pruned {beforePruneSize - afterPruneSize} contours")

                #print(f"Contained contours after prune have sizes: {[contour.size for contour in containedContours]}")
                #print(f"Number of contained contours after prune: {len(containedContours)}")

            
            if len(containedContours) == 1:
                #consume the active contour
                if containedContours[0].size > maxSize:
                    finalContours.append(containedContours[0])
                    contour.addChild(containedContours[0])
                    #print(f"Above max size, Contour {contour.index} added {containedContours[0].index} as child")
                else:
                    contour.consume(containedContours[0])
                    #print(f"Contour {contour.index} consumed {containedContours[0].index}")
            elif len(containedContours) > 1:
                #move contained contours to finalContours
                contour.addChildren(containedContours)

                finalContours += containedContours
                #print(f"Contour {contour.index} added {len(containedContours)} children")


        


        #drawImage = image.copy()
        #for contour in contours:
        #    cv2.drawContours(drawImage, [contour.contour], 0, (255, 255, 255), 1)
        #
        #cv2.imshow("Thresholds at intensity", drawImage)
        #key = cv2.waitKey(0)
        ## If the user presses 'q', exit the loop
        #if key == ord('q'):
        #    break


        activeContours = contours

    finalContours += activeContours

    #sort out those wo have grand children
    #finalContours = [contour for contour in finalContours if not contour.hasGrandChildren()]


    #


    


    #draw the tree
    drawImage = image.copy()
    #to color
    drawImage = cv2.cvtColor(drawImage, cv2.COLOR_GRAY2BGR)
    for contour in finalContours:

        #color depeds on whether it has children, grand children, or not
        if contour.hasGrandChildren():
            color = (0, 255, 0) #green
        elif contour.hasChildren():
            color = (0, 0, 255) #Red
        else:
            color = (255, 0, 0) #Blue



        cv2.drawContours(drawImage, [contour.contour], 0, color, 1)

        if contour.hasParent():
            cv2.line(drawImage, contour.center, contour.parent.center, (255, 255, 255), 1)

    cv2.imshow("Threshold Tree", drawImage)

    finalContours = [contour for contour in finalContours if not contour.hasGrandChildren()]

    #resize contours
    i = 0
    for contour in finalContours:
        pass
        contour.contour = contour.contour * downscaleFactor
        contour.index = i
        i += 1



    
    return finalContours


            

    #draw the contours
    #drawImage = image.copy()
    #for contour in finalContours:
    #    cv2.drawContours(drawImage, [contour.contour], 0, (255, 255, 255), 1)
#
    #cv2.imshow("Threshold Tree", drawImage)
            
                    


                    

            


    pass



def getSimilarity(contour1, contour2):

    #use shapely to find the intersection of the two contours. similarity is the area of the intersection divided by the total area of the two contours

    #convert the contours to shapely polygons
    
    similarity = 0
    try:

        poly1 = contourToPolygon(contour1)
        poly2 = contourToPolygon(contour2)

        #if either of the polygons are None, return 0
        if poly1 is None or poly2 is None:
            return 0


        #find the intersection of the two contours

        intersection = poly1.intersection(poly2)

        #find the area of the intersection
        intersectionArea = intersection.area

        #find the total area of the two contours
        totalArea = poly1.area + poly2.area - intersectionArea

        #find the similarity

        similarity = intersectionArea / totalArea if totalArea > 0 else 0
    except Exception as e:
        #print(f"Error: {e}")
        similarity = 0





    return similarity

def contourToPolygon(contour1):
    newContour1 = []
    for index, point in enumerate(contour1):
        newContour1.append((int(point[0][0]), int(point[0][1])))

    #print(newContour1)

    if len(newContour1) < 3:
        return None
    return Polygon(newContour1)

def matchContour(contour, contours):
    
    bestSimilarity = 0
    bestIndex = -1

    for index, otherContour in enumerate(contours):
        similarity = getSimilarity(contour, otherContour)
        if similarity > bestSimilarity:
            bestSimilarity = similarity
            bestIndex = index
    
    return bestIndex


if __name__ == '__main__':
    


    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")


    episodeManager.nextEpisode()
    #episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    for index in range(len(episode)):
        

        if index < 127:
            continue

        frame = episode[index]

        print(f"Processing frame {index}")
        #print(frame.image.shape)
        #findBranches(frame.image, doDraw=True)

        #watershed(frame.image)
        #watershed(frame.image)



        timeStamp1 = time.time()
        contours = thresholdTree(frame.image, 1)
        timeStamp2 = time.time()

        #time taken in seconds
        print(f"Time taken: {timeStamp2 - timeStamp1}")

        drawImage = frame.image.copy()
        for contour in contours:
            cv2.drawContours(drawImage, [contour.contour], 0, (255, 255, 255), 1)
        
        cv2.imshow("Threshold Contours", drawImage)

        #create_and_display_contour_tree(frame.image)
        
        key = cv2.waitKey(0)
        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
        # if esc is pressed, exit the loop
        elif key == 27:
            break


        

    cv2.destroyAllWindows()

