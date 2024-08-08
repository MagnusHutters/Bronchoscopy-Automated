

from DataHandling.Episode import EpisodeManager, Episode

import numpy as np
import cv2
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment

#import matplotlib.pyplot as plt



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


                #child.contour = self.contour
                #child.area = self.area
                #child.circumference = self.circumference
                #child.roundness = self.roundness

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
    trim=5
    iterations = 3
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

            contour = subnode.contour
            area = subnode.area
            circumference = subnode.circumference
            roundness = subnode.roundness
            center = point.coords

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

    
    






if __name__ == '__main__':
    


    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")


    episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    for frame, index in episode:
        if index < 150:
            continue

        print(f"Processing frame {index}")
        print(frame.image.shape)
        findBranches(frame.image, doDraw=True)

        
        key = cv2.waitKey(1)
        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
        # if esc is pressed, exit the loop
        elif key == 27:
            break


        

    cv2.destroyAllWindows()

