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
    
    annotations_path = "Photo_Interpretation_Data_jpg/Test/process_results.json"
    images_path = "Photo_Interpretation_Data_jpg/Test/Images"
    
    # lista_jpg = glob.glob(f"{images_path}/*.jpg")
    # lista_jpg = {i:j.split("/")[-1] for i,j in enumerate(sorted(lista_jpg,key=extract_number))}

    # Load COCO JSON
    with open(annotations_path) as f:
        coco = json.load(f)

    # Pick a random image
    image_info = random.choice(coco)
    image_id = image_info['image_id'] 
    # imagen_path = lista_jpg[image_info['image_id']]
    img_path = os.path.join(images_path, f"Img_{image_id}.jpg")

    # Load image
    image = Image.open(img_path)

    # Get annotations for this image
    anns = [ann for ann in coco if ann["image_id"] == image_id if ann['score'] > 0.3]

    # Plot
    _, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(image)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        # Optional: segmentation polygon (if present)
        if "segmentation" in ann:
            for seg in ann["segmentation"]:
                xs = seg[::2]
                ys = seg[1::2]
                ax.plot(
                    xs + [xs[0]], ys + [ys[0]], linestyle="-", linewidth=1.5, color="cyan"
                )

    ax.set_title(f"Image: {img_path} (id={image_id})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()