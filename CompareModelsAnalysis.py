




import numpy as np
import os
import cv2
import random
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, ttest_rel
from statsmodels.stats.contingency_tables import mcnemar

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
    
    return TP, FP, FN, matched_gt, matched_pred



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



def z_test_two_proportions(p1, n1, p2, n2):
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return z_score, p_value

def main():
    n=200
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
    b, c = 0, 0  # Initialize counts for the contingency table
    b2, c2 = 0, 0  # Initialize counts for the contingency table
    totalErrorsCV = []
    totalErrorsCNN = []

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


        
    
        tp_CNN, fp_CNN, fn_CNN, matched_gt_CNN, matched_pred_CNN = calculate_metrics(label, predictionNN, tolerance_radius)
        tp_CV, fp_CV, fn_CV,  matched_gt_CV, matched_pred_CV  = calculate_metrics(label, predictionCV, tolerance_radius)
        
        totalErrorsCNN.append(fp_CNN + fn_CNN)
        totalErrorsCV.append(fp_CV + fn_CV)

        TP_CNN += tp_CNN
        FP_CNN += fp_CNN
        FN_CNN += fn_CNN
        
        TP_CV += tp_CV
        FP_CV += fp_CV
        FN_CV += fn_CV


        set_matched_pred_CNN = set(matched_pred_CNN)
        set_matched_pred_CV = set(matched_pred_CV)
        
        b += len(set_matched_pred_CV - set_matched_pred_CNN)  # Correct in CV, incorrect in CNN
        c += len(set_matched_pred_CNN - set_matched_pred_CV)  # Correct in CNN, incorrect in CV
    

        set_matched_gt_CNN = set(matched_gt_CNN)
        set_matched_gt_CV = set(matched_gt_CV)

        b2 += len(set_matched_gt_CV - set_matched_gt_CNN)  # Correct in CV, incorrect in CNN
        c2 += len(set_matched_gt_CNN - set_matched_gt_CV)  # Correct in CNN, incorrect in CV



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

    CNN_N_Predictions = TP_CNN + FP_CNN
    CV_N_Predictions = TP_CV + FP_CV
    CNN_Precision = metrics_CNN['precision']
    CV_Precision = metrics_CV['precision']
    z_score, p_value = z_test_two_proportions(CNN_Precision, CNN_N_Predictions, CV_Precision, CV_N_Predictions)
    print(f"Z-Test Two Proportions: Z-Score: {z_score}, P-Value: {p_value}")
    # Interpretation of p-value
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between the precision of the two models.")
    else:
        print("There is no significant difference between the precision of the two models.")


    print("")
    t_stat, p_value = ttest_rel(totalErrorsCNN, totalErrorsCV)

    print(f'Paired t-test statistic: {t_stat}, P-value: {p_value}')
    print("")


    #contingency_table = [[0, b], [c, 0]]
    #contingency_table2 = [[0, b2], [c2, 0]]
    #print("Contingency Table 1:", contingency_table)
    #print("Contingency Table 2:", contingency_table2)

    #result = mcnemar(contingency_table, exact=True)
    #result2 = mcnemar(contingency_table2, exact=True)
    

    print("CNN Model Aggregate Metrics:", metrics_CNN)
    print("CV Model Aggregate Metrics:", metrics_CV)


    return
    print(f'McNemar test statistic: {result.statistic}')
    print(f'P-value: {result.pvalue}')

    # Interpretation of p-value
    if result.pvalue < 0.05:
        print("There is a significant difference between the performance of the two models.")
    else:
        print("There is no significant difference between the performance of the two models.")



    print(f'McNemar test statistic 2: {result2.statistic}')
    print(f'P-value 2: {result2.pvalue}')

    # Interpretation of p-value
    if result2.pvalue < 0.05:
        print("There is a significant difference between the performance of the two models.")
    else:
        print("There is no significant difference between the performance of the two models.")



        








if __name__ == '__main__':
    main()