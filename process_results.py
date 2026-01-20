import torchvision
import glob
import re
import os
import torch
import json

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1

def main():
    keep_all_indices : bool = True
    extension = ""
    annotations_path = f"Photo_Interpretation_Data{extension}/Test/codetr_output.bbox.json"
    out_path = f"Photo_Interpretation_Data{extension}/Test/process_results.json"
    images_path = f"Photo_Interpretation_Data{extension}/Test/Images"
    
    if extension == "_jpg": 
        lista_jpg = glob.glob(f"{images_path}/*.jpg")
    else:
        lista_jpg = glob.glob(f"{images_path}/*.tif*")
    print(lista_jpg)
    lista_jpg = {i:extract_number(j.split("/")[-1]) for i,j in enumerate(sorted(lista_jpg,key=extract_number))}

    # Load COCO JSON
    with open(annotations_path) as f:
        coco = json.load(f)

    ids = []
    boxes = []
    scores = []
    new_json = []
    for i in range(0,len(coco)):
        image_info = coco[i]
        image_id = lista_jpg[image_info['image_id']]
        print(image_id)
        score = image_info['score']
        box = image_info['bbox']
        
        ids.append(image_id)
        boxes.append(box)
        scores.append(score)
        
        new_json.append({'image_id':image_id, 'bbox':box, 'score':score , 'category':1})

    print(scores)
    #  Compatible tensor type
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Convert XYWH → XYXY
    def xywh_to_xyxy(boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x, y, x + w, y + h], dim=1)

    boxes = xywh_to_xyxy(boxes)
    if not keep_all_indices:
        # NMS  -> merge similar bbox
        keep_indices = torchvision.ops.nms(boxes, scores, 0.5)         # (N,)
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
            
        keep = keep_indices.tolist()     
        new_json = [new_json[i] for i in keep] 
    
    with open(out_path, "w") as f:
        json.dump(new_json, f, indent=4)


if __name__ == "__main__":
    main()