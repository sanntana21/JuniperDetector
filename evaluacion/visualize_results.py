"""Dibuja sobre una imagen las predicciones del modelo junto a las anotaciones de referencia.

Toma una imagen al azar del fichero de predicciones procesadas del test fotointerpretado o
del test de trabajo de campo y superpone en rojo las predicciones que superan una confianza
mínima y en azul las anotaciones. Sirve para la inspección cualitativa de los resultados:
muestra la figura por pantalla y no guarda ningún fichero.
"""

import json
import glob
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1

mode_map = {0: "Train", 1: "Val", 2: "Test", 3: "FieldWork"}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="Modo: 0 = PW, 1 = FW",
    )
    
    args = parser.parse_args()
    mode = args.mode
    extension = ""
    segmentation = False
    output_type = "_segm" if segmentation else "_bbox"
    
        
    if mode == 1:
        og_path = f"Field_Work_Data{extension}/External_Val_Data/Annotations/FieldWork_updated.json"
        annotations_path = f"Field_Work_Data{extension}/External_Val_Data/process_results{output_type}.json"
        images_path = f"Field_Work_Data{extension}/External_Val_Data/Images"
    else:
        og_path = f"Photo_Interpretation_Data{extension}/Test/Annotations/Test_updated.json"
        annotations_path = f"Photo_Interpretation_Data{extension}/Test/process_results{output_type}.json"
        images_path = f"Photo_Interpretation_Data{extension}/Test/Images"
    
    lista_jpg = glob.glob(f"{images_path}/*.tif")
    lista_jpg = {i:j.split("/")[-1] for i,j in enumerate(sorted(lista_jpg,key=extract_number))}
    print(lista_jpg)

    # Load COCO JSON
    with open(og_path) as f:
        og_coco = json.load(f)
        og_coco = og_coco['annotations']
        
    for a in og_coco:
        a['image_id'] = int(a['image_id'])
        a['file_name'] = lista_jpg[int(a['image_id'])]
        
    with open(annotations_path) as f:
        coco = json.load(f)

    # Pick a random image
    image_info = random.choice(coco)
    image_id = image_info['image_id']
    
    img_path = os.path.join(images_path, f"Img_{image_id}.tif")

    # Load image
    image = Image.open(img_path)
    
    # Get annotations for this image
    print(image_id)
    og_anns = [ann for ann in og_coco if ann["file_name"] == f"Img_{image_id}.tif"]
    anns = [ann for ann in coco if ann["image_id"] == image_id and ann['score'] > 0.2]
    
    # Plot
    _, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(image)
    
    for ann in anns:
        if "segmentation" in ann:
            polygons = ann["segmentation"]
            
            for poly in polygons:
                xs = poly[::2]
                ys = poly[1::2]
                ax.plot(
                    xs + [xs[0]], ys + [ys[0]],
                    linestyle="-", linewidth=1.5, color="red"
                ) 
        else:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            
    for ann in og_anns:
        if "segmentation" in ann:
            for seg in ann["segmentation"]:
                xs = seg[::2]
                ys = seg[1::2]
                ax.plot(
                    xs + [xs[0]], ys + [ys[0]],
                    linestyle="-", linewidth=1.5, color="blue"
                ) 
                
        else:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none"
            )
            ax.add_patch(rect)

    ax.set_title(f"Image: {img_path} (id={image_id})")
    plt.axis("off")
    plt.show()
        
    # else:
    #     print(og_anns,"here")
    #     for ann in anns:
    #         x, y, w, h = ann["bbox"]
    #         rect = patches.Rectangle(
    #             (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
    #         )
    #         ax.add_patch(rect)

    #         # Optional: segmentation polygon (if present)
    #         if "segmentation" in ann:
    #             for seg in ann["segmentation"]:
    #                 xs = seg[::2]
    #                 ys = seg[1::2]
    #                 ax.plot(
    #                     xs + [xs[0]], ys + [ys[0]], linestyle="-", linewidth=1.5, color="cyan"
    #                 )
        
    #     og_anns = [ann for ann in og_coco if ann["image_id"] == int(image_id)]
    #     for ann in og_anns:
    #         x, y, w, h = ann["bbox"]
    #         rect = patches.Rectangle(
    #             (x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none"
    #         )
    #         ax.add_patch(rect)

    #         # Optional: segmentation polygon (if present)
    #         if "segmentation" in ann:
    #             for seg in ann["segmentation"]:
    #                 #This is the segmentation I expect to have
    #                 xs = seg[::2]
    #                 ys = seg[1::2]
    #                 ax.plot(
    #                     xs + [xs[0]], ys + [ys[0]], linestyle="-", linewidth=1.5, color="cyan"
    #                 )
        

    #     ax.set_title(f"Image: {img_path} (id={image_id})")
    #     plt.axis("off")
    #     plt.show()

if __name__ == "__main__":
    main()