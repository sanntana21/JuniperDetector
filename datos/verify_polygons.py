"""Dibuja los polígonos YOLO-seg sobre sus imágenes para comprobar visualmente la conversión.

Lee un split del dataset generado por prepare_dataset_segmentation.py y guarda en
verified_polygons_<split>/ un PNG por imagen con los contornos y su clase dibujados, más un collage
_grid.png opcional, e imprime el área en píxeles de cada polígono. Se apoya en polygon_utils.py y
replica el modo de uso de verify_boxes.py.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from polygon_utils import denormalize_polygon, polygon_area


def read_yolo_seg_labels(label_path: Path):
    if not label_path.exists():
        return []
    polys = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        polys.append((cls, coords))
    return polys


def draw_polygons(img, polys, class_names=None, color=(0, 0, 255), thickness=1):
    h, w = img.shape[:2]
    for cls, coords_norm in polys:
        coords_px = denormalize_polygon(coords_norm, w, h)
        pts = np.array(coords_px, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        x0, y0 = pts[0]
        cv2.putText(img, label, (int(x0), max(int(y0) - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def load_class_names(dataset_dir: Path):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return None
    names, in_names_block = [], False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names_block = True
            continue
        if in_names_block:
            if stripped and ":" in stripped and not stripped.startswith("#"):
                _, name = stripped.split(":", 1)
                names.append(name.strip())
            elif not stripped:
                break
    return names or None


def make_grid(images, cols=4, cell_size=250, pad=4):
    n = len(images)
    if n == 0:
        return None
    rows = (n + cols - 1) // cols
    grid_h = rows * (cell_size + pad) + pad
    grid_w = cols * (cell_size + pad) + pad
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for idx, (name, img) in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        y0 = pad + r * (cell_size + pad)
        x0 = pad + c * (cell_size + pad)
        grid[y0:y0 + cell_size, x0:x0 + cell_size] = resized
        cv2.putText(grid, name, (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return grid


def main():
    parser = argparse.ArgumentParser(description="Verifica visualmente poligonos YOLO-seg sobre las imagenes.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--sort", choices=["random", "densest", "sparsest"], default="random")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    out_dir = args.out or (args.dataset / f"verified_polygons_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise SystemExit(f"No existe la carpeta de imagenes: {img_dir}")

    class_names = load_class_names(args.dataset)
    all_images = sorted(img_dir.glob("*"))
    if not all_images:
        raise SystemExit(f"No se encontraron imagenes en {img_dir}")

    catalog = []
    for img_path in all_images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        polys = read_yolo_seg_labels(lbl_path)
        catalog.append((img_path, polys))

    if args.names:
        selected = [(p, b) for p, b in catalog if p.stem in set(args.names)]
    elif args.only_empty:
        selected = [(p, b) for p, b in catalog if len(b) == 0]
        print(f"Imagenes sin anotacion en '{args.split}': {len(selected)} / {len(catalog)}")
    elif args.all:
        selected = catalog
    else:
        if args.sort == "densest":
            selected = sorted(catalog, key=lambda x: -len(x[1]))[:args.n]
        elif args.sort == "sparsest":
            selected = sorted(catalog, key=lambda x: len(x[1]))[:args.n]
        else:
            random.seed(args.seed)
            selected = random.sample(catalog, min(args.n, len(catalog)))

    if not selected:
        print("No hay imagenes que cumplan el criterio seleccionado.")
        return

    print(f"\nRevisando {len(selected)} imagenes del split '{args.split}' -> {out_dir}\n")

    grid_images = []
    for img_path, polys in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [ERROR] No se pudo leer: {img_path}")
            continue
        drawn = draw_polygons(img.copy(), polys, class_names)
        out_path = out_dir / f"{img_path.stem}_verified.png"
        cv2.imwrite(str(out_path), drawn)

        areas = [polygon_area(denormalize_polygon(c, img.shape[1], img.shape[0])) for _, c in polys]
        area_info = f"areas px^2: {[round(a) for a in areas]}" if areas else ""
        print(f"  {img_path.name:<20} {len(polys):>3} poligono(s)  {area_info}")

        if args.grid:
            grid_images.append((img_path.stem, drawn))

    if args.grid and grid_images:
        grid = make_grid(grid_images)
        if grid is not None:
            grid_path = out_dir / "_grid.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"\nCollage generado en: {grid_path}")

    print(f"\nListo. Revisa las imagenes en: {out_dir}")


if __name__ == "__main__":
    main()
