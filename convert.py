import os
import json
import argparse


def calculate_bbox(xs, ys):
    min_x, min_y = min(xs), min(ys)
    width = max(xs) - min_x
    height = max(ys) - min_y
    return [min_x, min_y, width, height]


def convert_segmentations_to_annotations(ann, base_id):
    annotations = []
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    iscrowd = ann["iscrowd"]

    for i, seg in enumerate(ann["segmentation"]):
        xs = seg[::2]
        ys = seg[1::2]
        bbox = calculate_bbox(xs, ys)
        area = bbox[2] * bbox[3]

        annotations.append(
            {
                "id": base_id,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": iscrowd,
                "area": area,
                "bbox": bbox,
                "segmentation": [seg],
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            }
        )
        base_id += 1

    return annotations, base_id


mode_map = {
    0: "Train",
    1: "Val",
    2: "Test"
}

def main():
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode", type=int, choices=[0, 1, 2], default=0,
        help="Modo: 0 = Train (por defecto), 1 = Val, 2 = Test"
    )
    args = parser.parse_args()

    mode_str = mode_map[args.mode]
    print(f"Modo seleccionado: {mode_str}")

    # Lógica según modo
    if mode_str == "Train":
        print("Ejecutando entrenamiento...")
    elif mode_str == "Val":
        print("Ejecutando validación...")
    elif mode_str == "Test":
        print("Ejecutando prueba...")
    # --- Settings ---
    annotations_dir = f"./Photo_Interpretation_Data/{mode_str}/Annotations"
    images_dir = f"./Photo_Interpretation_Data/{mode_str}/Images"
    #shapefiles_dir = f"./Photo_Interpretation_Data/{model_str}/Annotations/Shapefiles"

    coco_path = os.path.join(annotations_dir, f"{mode_str}.json")

    # --- Load existing COCO JSON ---
    with open(coco_path) as f:
        coco = json.load(f)

    # Build index of image file names → image IDs
    filename_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    # --- Initialize ---
    new_annotations = []
    ann_id = 0
    anns = [ann for ann in coco["annotations"]]

    for ann in anns:
        trans_ann, ann_id = convert_segmentations_to_annotations(ann, ann_id)
        new_annotations.extend(trans_ann)

    coco["annotations"] = new_annotations

    out_path = os.path.join(annotations_dir, f"{mode_str}_updated.json")
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Saved updated COCO annotations to: {out_path}")
    print(f"➕ Added {len(new_annotations)} new grouped annotations.")

if __name__ == "__main__":
    main()