




import numpy as np
import os
import cv2
import random
from scipy.optimize import linear_sum_assignment

from Training.BasicPaths import *



from Training.CVPathsFinder import doCVPathFinding







def drawPoints(image, points, color=(0, 0, 1), radius=8, thickness=2):
    imageSize = image.shape[:2]
    for hole in points:
        x = int((hole[0] + 1) / 2 * imageSize[0])
        y = int((hole[1] + 1) / 2 * imageSize[1])
        cv2.circle(image, (x, y), radius, color, thickness)



def calculate_distance_matrix(ground_truth, predictions):
    distance_matrix = np.zeros((len(ground_truth), len(predictions)))
    for i, gt in enumerate(ground_truth):
        for j, pred in enumerate(predictions):
            distance_matrix[i, j] = np.linalg.norm(np.array(gt) - np.array(pred))
    return distance_matrix

def calculate_metrics(ground_truth, predictions, tolerance_radius):
    distance_matrix = calculate_distance_matrix(ground_truth, predictions)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    TP, FP, FN = 0, 0, 0
    matched_gt = set()
    matched_pred = set()
    
    for i, j in zip(row_ind, col_ind):
        if distance_matrix[i, j] <= tolerance_radius:
            TP += 1
            matched_gt.add(i)
            matched_pred.add(j)
    
    FN = len(ground_truth) - len(matched_gt)
    FP = len(predictions) - len(matched_pred)
    
    return TP, FP, FN




def compute_metrics(TP, FP, FN):
    #accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        #'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def main():
    n=1000
    tolerance_radius=0.25

    path = "Training/Data/PathData"

    #create the directory to save the images
    outputPath = "Output/ComparisonImages"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


    model = tf.keras.models.load_model("pathModelLabel.keras")

    images, realImageSize, originals = load_images(path, IMAGE_SIZE, saveOriginalImages=True)
    
    tolerance_radius_int = int(tolerance_radius * realImageSize[0] * 0.5)
    
    labels = load_labels(path, realImageSize[0], realImageSize[1])


    indices = list(range(len(labels)))

    # Randomly select n indices
    random.seed(42)
    selected_indices = random.sample(indices, n)



    images = images[selected_indices]
    labels = labels[selected_indices]
    originals = originals[selected_indices]

    print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
    TP_CNN, FP_CNN, FN_CNN = 0, 0, 0
    TP_CV, FP_CV, FN_CV = 0, 0, 0

    for i in range(n):
        #print(f"Image {i}")

        image = images[i]
        label = labels[i]
        displayImage = originals[i].copy()


        #print(label)




        predictionNN = model.predict(np.array([image]), verbose=0)[0]



        predictionCV = doCVPathFinding(image)


        #remove predictions with confidence less than confidenceValueMin, then remove the confidence value
        confidenceValueMin = 0.33
        predictionNN = [p[:2] for p in predictionNN if p[2] >= confidenceValueMin]
        predictionCV = [p[:2] for p in predictionCV if p[2] >= confidenceValueMin]
        label = [p[:2] for p in label if p[2] >= confidenceValueMin]

        #display the image with the predictions


        
    
        tp_CNN, fp_CNN, fn_CNN = calculate_metrics(label, predictionNN, tolerance_radius)
        tp_CV, fp_CV, fn_CV = calculate_metrics(label, predictionCV, tolerance_radius)
        
        TP_CNN += tp_CNN
        FP_CNN += fp_CNN
        FN_CNN += fn_CNN
        
        TP_CV += tp_CV
        FP_CV += fp_CV
        FN_CV += fn_CV




        drawPoints(displayImage, label, color=(0, 1, 0), radius = tolerance_radius_int, thickness=1)
        drawPoints(displayImage, label, color=(0, 1, 0), radius = 2, thickness=4)
        drawPoints(displayImage, predictionNN, color=(1, 0, 0), radius = 2, thickness=4)
        drawPoints(displayImage, predictionCV, color=(0, 0, 1), radius = 2, thickness=4)

        #put text on the image
        cv2.putText(displayImage, "Ground Truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 1, 0), 2)
        cv2.putText(displayImage, "CNN", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 0, 0), 2)
        cv2.putText(displayImage, "CV", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)

        #convert and save the image 
        displayImage = (displayImage * 255).astype(np.uint8)
        cv2.imwrite(f"{outputPath}/image{i:04d}.png", displayImage)


        #print(displayImage.shape)
#
        #cv2.imshow("DispImage", displayImage)
        #key = cv2.waitKey(0)
        ##quit on q or escape
        #if key == ord('q') or key == 27:
        #    break


    metrics_CNN = compute_metrics(TP_CNN, FP_CNN, FN_CNN)
    metrics_CV = compute_metrics(TP_CV, FP_CV, FN_CV)


    print("CNN Model Aggregate Metrics:", metrics_CNN)
    print("CV Model Aggregate Metrics:", metrics_CV)



        








if __name__ == '__main__':
    main()