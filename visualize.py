import json
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


mode_map = {0: "Train", 1: "Val", 2: "Test", 3: "FieldWork"}

def main():
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode", type=int, choices=[0, 1, 2, 3], default=0,
        help="Modo: 0 = Train (por defecto), 1 = Val, 2 = Test"
    )
    args = parser.parse_args()

    mode_str = mode_map[args.mode]
    print(f"Modo seleccionado: {mode_str}")

    # Lógica según modo
    if mode_str == "Train":
        print("Visualizado entrenamiento...")
    elif mode_str == "Val":
        print("Visualizado validación...")
    elif mode_str == "Test":
        print("Visualizado prueba...")
    elif mode_str == "FieldWork":
        print("Visualizando Field Work")
    else:
        raise ValueError("El modo seleccionado no es valido.")

    # --- Settings ---
    if args.mode < 3:
        annotations_path = f"./Photo_Interpretation_Data/{mode_str}/Annotations/{mode_str}_updated.json"
        images_path = f"./Photo_Interpretation_Data/{mode_str}/Images"
    else:
        annotations_path = "./Field_Work_Data/External_Val_Data/Annotations/Shapefiles/FieldWork.json"
        #annotations_path = "./Field_Work_Data/External_Val_Data/Prueba/FieldWork.json"
        images_path = "./Field_Work_Data/External_Val_Data/Images"

    # Load COCO JSON
    with open(annotations_path) as f:
        coco = json.load(f)

    # Pick a random image
    image_info = random.choice(coco["images"])
    img_path = os.path.join(images_path, image_info["file_name"])
    image_id = image_info["id"]

    # Load image
    image = Image.open(img_path)

    # Get annotations for this image
    anns = [ann for ann in coco["annotations"] if ann["image_id"] == image_id]

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

    ax.set_title(f"Image: {image_info['file_name']} (id={image_id})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()