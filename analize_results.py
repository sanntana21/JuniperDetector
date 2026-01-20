import json
import glob
import random
import os
import matplotlib.pyplot as plt

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from shapely.geometry import shape
import numpy as np
from utils import extract_number
from utils import xywh_to_xyxy
import matplotlib.pyplot as plt


def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def iou_single(boxA, boxB, eps=1e-9):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    return inter_area / (union + eps)

import numpy as np

def iou_matrix(pred_boxes, gt_boxes):
    P = len(pred_boxes)
    G = len(gt_boxes)
    M = np.zeros((P, G))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            M[i, j] = iou_single(p, g)
    return M

def match_greedy(iou_mat, iou_thr=0.5):
    M = iou_mat.copy()
    matched_pred = set()
    matched_gt = set()

    while True:
        i, j = np.unravel_index(np.argmax(M), M.shape)
        maxval = M[i, j]
        if maxval < iou_thr:
            break

        matched_pred.add(i)
        matched_gt.add(j)

        M[i, :] = -1
        M[:, j] = -1

    return matched_pred, matched_gt


mode_map = {0: "Test", 1: "FieldWork"}

def run_analysis(
    confidence_threshold : float = 0.5, 
    iou_thr : float = 0.5, 
    use_siou: bool = False, 
    extension  : str = "", 
    verbose : int = 0
):
    if verbose > 0:
        print(f"Ejecutando analisis = {confidence_values}, iou_thr = {iou_thr} , extension = {extension}")
        
    photo_interpretation_test_dir = f"Photo_Interpretation_Data{extension}/Test"
    # annotations_path = f"Photo_Interpretation_Data{extension}/Test/codetr_output.bbox.json"
    results_path = f"{photo_interpretation_test_dir}/process_results.json"
    original_path = f"{photo_interpretation_test_dir}/Annotations/Test_updated.json"
    
    # results_path = annotations_path
    
    # Load COCO JSON
    with open(original_path) as f:
        coco_original = json.load(f)

    with open(results_path) as f:
        coco_results = json.load(f)
    
    coco_results = [a for a in coco_results if a.get("score") > confidence_threshold]
    
    anns = [
        {
            **{k:v for k,v in ann.items() if k != 'segmentation'},
            "id": extract_number(coco_original["images"][ann["image_id"]]["file_name"])
        }
        for ann in coco_original["annotations"]
    ]
    
    catalog = list(set([a.get("id") for a in anns]))
    
    TP_total = 0
    FP_total = 0 
    FN_total = 0 
   
    for img_id in catalog:
        
    
        if verbose > 1:
            print("--------------------------------------\n")
            print([a for a in coco_results if a.get("image_id") < 5])
            print(coco_results)
            
        original_id_anns = [
            a for a in anns if a.get("id") == img_id
        ] 
        predicted_id_anns  = [
            a for a in coco_results if a.get("image_id") == img_id
        ]
        
        original_id_anns   = [xywh_to_xyxy(b.get("bbox")) for b in original_id_anns]
        predicted_id_anns = [xywh_to_xyxy(b.get("bbox")) for b in predicted_id_anns]
         
        # M = iou_matrix(original_id_anns, predicted_id_anns)
        M = iou_matrix(predicted_id_anns, original_id_anns)

        if verbose > 0:
            print("image_id", img_id)
            print("original id anss", original_id_anns)
            print("Predicted id:",predicted_id_anns)
            print("IoU matrix:\n", np.round(M, 3))
       
        if predicted_id_anns: 
            matched_pred, matched_gt = match_greedy(M, iou_thr=iou_thr)

            TP = len(matched_pred)
            FP = len(predicted_id_anns) - TP
            FN = len(original_id_anns) - TP
        else:
            TP = 0
            FP = 0
            FN = len(original_id_anns)

        print("TP:", TP, "FP:", FP, "FN:", FN)
            
        TP_total += TP
        FP_total += FP 
        FN_total += FN 
        
    den_total = 2 * TP_total + FP_total + FN_total
    F1_total = (2 * TP_total / den_total) if den_total > 0 else 0.0

    if verbose > 1:
        print("TOTAL:")
        print("TP:", TP_total, "FP:", FP_total, "FN:", FN_total)
        print("F1 TOTAL:", F1_total) 
    
    return F1_total, TP_total, FP_total, FN_total
    
if __name__ == "__main__":
    results_F1 = [] 
    results_TP = [] 
    results_FP = [] 
    results_FN = []
    confidence_values = [i / 10 for i in range(1, 10)]
    
    for confidence in confidence_values:
        F1,TP,FP,FN = run_analysis(confidence_threshold=confidence,iou_thr=0.1)
        results_F1.append(F1)
        results_TP.append(TP)
        results_FP.append(FP)
        results_FN.append(FN)
        
    plt.figure()
    plt.plot(confidence_values, results_F1, marker='o')
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 score")
    plt.title("F1 score vs Confidence threshold")
    plt.grid(True)
    plt.show()