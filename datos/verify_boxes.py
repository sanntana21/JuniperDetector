"""Dibuja las cajas YOLO sobre sus imágenes para comprobar visualmente la conversión.

Lee un split del dataset generado por prepare_dataset.py, selecciona imágenes al azar, por densidad
de cajas, por nombre o todas, y guarda en verified_<split>/ un PNG por imagen con las cajas y su
clase superpuestas, más un collage _grid.png si se pide. Sirve de control de calidad de las
etiquetas antes de entrenar.
"""

import argparse
import random
from pathlib import Path

import cv2


def read_yolo_labels(label_path: Path):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        boxes.append((cls, xc, yc, bw, bh))
    return boxes


def draw_boxes(img, boxes, class_names=None, color=(0, 0, 255), thickness=1):
    h, w = img.shape[:2]
    for cls, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        cv2.putText(img, label, (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def load_class_names(dataset_dir: Path):
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        return None
    names = []
    in_names_block = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names_block = True
            continue
        if in_names_block:
            if stripped and ":" in stripped:
                # formato "  0: Juniperus"
                _, name = stripped.split(":", 1)
                names.append(name.strip())
            elif not stripped:
                break
    return names or None


def make_grid(images, cols=4, cell_size=250, pad=4):
    """Combina varias imagenes en una sola grid para revisar de un vistazo."""
    n = len(images)
    if n == 0:
        return None
    rows = (n + cols - 1) // cols
    grid_h = rows * (cell_size + pad) + pad
    grid_w = cols * (cell_size + pad) + pad
    grid = 255 * (cv2.UMat(grid_h, grid_w, cv2.CV_8UC3).get() * 0 + 1)

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
    parser = argparse.ArgumentParser(description="Verifica visualmente cajas YOLO sobre las imagenes.")
    parser.add_argument("--dataset", required=True, type=Path, help="Carpeta raiz del dataset YOLO")
    parser.add_argument("--split", default="val",
                         help="Nombre del split a revisar: train, val, test, field_work, "
                              "o cualquier otro split extra presente en el dataset (default: val)")
    parser.add_argument("--n", type=int, default=12, help="Numero de imagenes a revisar (default: 12)")
    parser.add_argument("--names", nargs="+", default=None,
                         help="Nombres concretos de imagen (sin extension) a revisar, ignora --n")
    parser.add_argument("--all", action="store_true", help="Revisar todas las imagenes del split")
    parser.add_argument("--only-empty", action="store_true",
                         help="Revisar solo imagenes SIN ninguna anotacion")
    parser.add_argument("--sort", choices=["random", "densest", "sparsest"], default="random",
                         help="Como elegir las imagenes cuando no se usa --names ni --all")
    parser.add_argument("--out", type=Path, default=None,
                         help="Carpeta de salida (default: dataset/verified_<split>)")
    parser.add_argument("--grid", action="store_true",
                         help="Ademas de guardar imagenes individuales, genera un collage grid.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    out_dir = args.out or (args.dataset / f"verified_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise SystemExit(f"No existe la carpeta de imagenes: {img_dir}")

    class_names = load_class_names(args.dataset)

    all_images = sorted(img_dir.glob("*"))
    if not all_images:
        raise SystemExit(f"No se encontraron imagenes en {img_dir}")

    # construir lista (stem, n_boxes) para poder ordenar/filtrar
    catalog = []
    for img_path in all_images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = read_yolo_labels(lbl_path)
        catalog.append((img_path, boxes))

    if args.names:
        selected = [(p, b) for p, b in catalog if p.stem in set(args.names)]
        missing = set(args.names) - {p.stem for p, _ in selected}
        if missing:
            print(f"[AVISO] No se encontraron estas imagenes: {sorted(missing)}")
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
    for img_path, boxes in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [ERROR] No se pudo leer: {img_path}")
            continue
        drawn = draw_boxes(img.copy(), boxes, class_names)
        out_path = out_dir / f"{img_path.stem}_verified.png"
        cv2.imwrite(str(out_path), drawn)
        print(f"  {img_path.name:<20} {len(boxes):>3} caja(s) -> {out_path.name}")

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