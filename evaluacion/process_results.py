"""Prepara las predicciones crudas del detector para la evaluación.

Lee la salida COCO del modelo, sustituye el índice interno de imagen por el número real del
fichero, convierte las máscaras RLE a polígono en el flujo de segmentación y aplica
supresión de no máximos por imagen a las cajas. Escribe process_results_bbox.json o
process_results_segm.json junto a los datos del conjunto correspondiente.
"""

import torchvision
import glob
import re
import os
import torch
import json
import argparse
from collections import defaultdict

from skimage import measure
from pycocotools import mask as mask_utils
import numpy as np
from scipy.ndimage import binary_dilation

def remove_container_boxes(boxes,
                           contain_thr=0.95,
                           area_ratio=1.5):
    """
    Elimina cajas que contienen casi completamente a otras.
    Siempre conserva la caja más pequeña.
    """

    keep = torch.ones(len(boxes), dtype=torch.bool)

    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])

    for i in range(len(boxes)):
        if not keep[i]:
            continue

        for j in range(i+1, len(boxes)):
            if not keep[j]:
                continue

            xx1 = max(boxes[i,0], boxes[j,0])
            yy1 = max(boxes[i,1], boxes[j,1])
            xx2 = min(boxes[i,2], boxes[j,2])
            yy2 = min(boxes[i,3], boxes[j,3])

            inter_w = max(0.0, xx2-xx1)
            inter_h = max(0.0, yy2-yy1)
            inter = inter_w * inter_h

            if inter == 0:
                continue

            # ¿i contiene casi completamente a j?
            contain_i = inter / areas[j]

            # ¿j contiene casi completamente a i?
            contain_j = inter / areas[i]

            if contain_i >= contain_thr and areas[i] > area_ratio * areas[j]:
                keep[i] = False      # eliminar la grande

            elif contain_j >= contain_thr and areas[j] > area_ratio * areas[i]:
                keep[j] = False      # eliminar la grande

    return torch.where(keep)[0]

def rle_to_polygon(rle_seg, dilate=True, dilation_iters=3):
    binary_mask = mask_utils.decode(rle_seg).astype(bool)
    
    if dilate:
        binary_mask = binary_dilation(binary_mask, iterations=dilation_iters)
    
    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
    
    if not contours:
        return []
    
    largest = max(contours, key=len)
    largest = np.fliplr(largest)
    return [largest.flatten().tolist()]

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1

def main(): 
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="Modo: 0 = Photo_Interpretation_Data, 1 = FieldWorkData",
    )
    args = parser.parse_args()
    mode = args.mode
    keep_all_indices : bool = False
    extension = ""
    model = "swimt" 
    segmentation = False
    
    
    output_type = "segm" if segmentation else "bbox"
    
    if mode == 1:
        annotations_path = f"Field_Work_Data{extension}/External_Val_Data/{model}_output.{output_type}.json"
        out_path = f"Field_Work_Data{extension}/External_Val_Data/process_results_{output_type}.json"
        images_path = f"Field_Work_Data{extension}/External_Val_Data/Images"
    else:
        annotations_path = f"Photo_Interpretation_Data{extension}/Test/{model}_output.{output_type}.json"
        out_path = f"Photo_Interpretation_Data{extension}/Test/process_results_{output_type}.json"
        images_path = f"Photo_Interpretation_Data{extension}/Test/Images"
        
    
    if extension == "_jpg": 
        lista_jpg = glob.glob(f"{images_path}/*.jpg")
    else:
        lista_jpg = glob.glob(f"{images_path}/*.tif*")
        
    lista_jpg = {i:extract_number(j.split("/")[-1]) for i,j in enumerate(sorted(lista_jpg,key=extract_number))}
    print(lista_jpg)
    
    # Load COCO JSON
    with open(annotations_path) as f:
        coco = json.load(f)

    ids = []
    boxes = []
    scores = []
    segs = []
    new_json = []
    for i in range(0,len(coco)):
        image_info = coco[i]
        image_id = lista_jpg[image_info['image_id']]
        score = image_info['score']
        
        ids.append(image_id)
        scores.append(score)
        
        if segmentation:
            seg = image_info['segmentation']
            segs.append(seg)
            new_json.append({'image_id': int(image_id), 'segmentation': seg, 'score': score, 'category': 1})
        else:        
            box = image_info['bbox']
            boxes.append(box)
            new_json.append({'image_id':int(image_id), 'bbox':box, 'score':score , 'category':1})

    if segmentation:

        new_json_decoded = []
        for ann in new_json:
            seg = ann['segmentation']
            new_json_decoded.append({
                'image_id': ann['image_id'],
                'segmentation': rle_to_polygon(seg),
                'score': ann['score'],
                'category': ann['category']
            })
        new_json = new_json_decoded
        
    else:
        #  Compatible tensor type
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Convert XYWH → XYXY
        def xywh_to_xyxy(boxes):
            x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            return torch.stack([x, y, x + w, y + h], dim=1)

        boxes = xywh_to_xyxy(boxes)
        if not keep_all_indices:
            from collections import defaultdict

            grouped = defaultdict(list)

            for ann in new_json:
                grouped[ann["image_id"]].append(ann)

            filtered_json = []

            for image_id, anns in grouped.items():

                boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
                scores = torch.tensor([a["score"] for a in anns], dtype=torch.float32)

                boxes = xywh_to_xyxy(boxes)
                
                # Solo elimina cajas prácticamente idénticas
                keep = torchvision.ops.nms(
                    boxes,
                    scores,
                    iou_threshold=0.9   # puedes probar 0.90–0.98
                )

                filtered_json.extend([anns[i] for i in keep.tolist()])

            new_json = filtered_json
            # # NMS  -> merge similar bbox
            # keep_indices = torchvision.ops.nms(boxes, scores, 0.5)         # (N,)
            # keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
                
            # keep = keep_indices.tolist()     
            # new_json = [new_json[i] for i in keep] 
    
    with open(out_path, "w") as f:
        json.dump(new_json, f, indent=4)
        
    print(f"GENERADO: {out_path}")


if __name__ == "__main__":
    main()