"""Genera tres subconjuntos YOLO por tamaño de caja a partir de un dataset ya convertido.

Lee el dataset YOLO de prepare_dataset.py, calcula sobre el split de entrenamiento los cuantiles
del área de las cajas y escribe tres copias del dataset (<prefijo>_small, <prefijo>_medium y
<prefijo>_large) en las que cada etiqueta solo aparece si su área cae en el rango correspondiente,
junto al resumen <prefijo>_size_buckets_summary.json. Prepara los datos de los expertos por tamaño.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def get_image_size(img_path: Path):
    """Devuelve (width, height) reales de una imagen, usando PIL o
    OpenCV segun lo que este disponible."""
    if _HAS_PIL:
        with Image.open(img_path) as im:
            return im.size  # (w, h)
    if _HAS_CV2:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        return w, h
    raise SystemExit("Necesito PIL (Pillow) u OpenCV instalado para leer el tamano de las imagenes.")


def read_data_yaml_names(data_yaml_path: Path):
    """Parseo simple del bloque 'names:' de un data.yaml generado por
    prepare_dataset.py (formato 'idx: nombre', sin depender de pyyaml)."""
    names = []
    in_names = False
    for line in data_yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if stripped.startswith("#") or not stripped:
                if not stripped:
                    break
                continue
            if ":" in stripped:
                _, name = stripped.split(":", 1)
                names.append(name.strip())
            else:
                break
    return names or ["object"]


def find_image_for_label(images_dir: Path, stem: str):
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def collect_bbox_areas(dataset_root: Path, split: str):
    """Devuelve una lista de (label_path, line_idx, area_px, line_str)
    para todas las cajas de un split, en area de PIXELES reales."""
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split
    if not labels_dir.exists():
        return []

    entries = []
    img_size_cache = {}

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        if stem not in img_size_cache:
            img_path = find_image_for_label(images_dir, stem)
            if img_path is None:
                print(f"  [AVISO] no se encuentra la imagen de {label_path.name}, se omite")
                img_size_cache[stem] = None
                continue
            img_size_cache[stem] = get_image_size(img_path)

        size = img_size_cache[stem]
        if size is None:
            continue
        w, h = size

        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.split()
            cls, xc, yc, bw, bh = parts[0], *map(float, parts[1:5])
            area_px = (bw * w) * (bh * h)
            entries.append((label_path, i, area_px, line))

    return entries


def compute_quantiles(areas, q_low, q_high):
    areas_sorted = sorted(areas)
    n = len(areas_sorted)
    if n == 0:
        raise SystemExit("No se encontraron bboxes en el split de train -- no se pueden calcular cuantiles.")

    def pct(q):
        idx = min(n - 1, max(0, int(round(q * (n - 1)))))
        return areas_sorted[idx]

    return pct(q_low), pct(q_high)


def bucket_for_area(area_px, low_thr, high_thr):
    if area_px <= low_thr:
        return "small"
    if area_px <= high_thr:
        return "medium"
    return "large"


def write_bucketed_dataset(dataset_root: Path, output_prefix: Path, bucket: str,
                            splits, low_thr, high_thr, names, use_symlink=True):
    out_root = Path(f"{output_prefix}_{bucket}")
    out_images = out_root / "images"
    out_labels_base = out_root / "labels"
    out_root.mkdir(parents=True, exist_ok=True)
    out_labels_base.mkdir(parents=True, exist_ok=True)

    # 'images' se comparte tal cual (mismas imagenes en los 3 buckets) via symlink
    # de la carpeta entera, para no triplicar el dataset en disco.
    if not out_images.exists():
        if use_symlink:
            out_images.symlink_to((dataset_root / "images").resolve())
        else:
            import shutil
            shutil.copytree(dataset_root / "images", out_images)

    stats = {}
    for split in splits:
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        if not labels_dir.exists():
            continue

        out_labels_split = out_labels_base / split
        out_labels_split.mkdir(parents=True, exist_ok=True)

        n_imgs, n_boxes_kept, n_boxes_total, n_imgs_empty = 0, 0, 0, 0
        img_size_cache = {}

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            if stem not in img_size_cache:
                img_path = find_image_for_label(images_dir, stem)
                img_size_cache[stem] = get_image_size(img_path) if img_path else None
            size = img_size_cache[stem]

            kept_lines = []
            if size is not None:
                w, h = size
                lines = label_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    if not line.strip():
                        continue
                    n_boxes_total += 1
                    parts = line.split()
                    cls, xc, yc, bw, bh = parts[0], *map(float, parts[1:5])
                    area_px = (bw * w) * (bh * h)
                    if bucket_for_area(area_px, low_thr, high_thr) == bucket:
                        kept_lines.append(line)

            (out_labels_split / label_path.name).write_text("\n".join(kept_lines), encoding="utf-8")
            n_imgs += 1
            n_boxes_kept += len(kept_lines)
            if not kept_lines:
                n_imgs_empty += 1

        stats[split] = {
            "n_images": n_imgs,
            "n_boxes_kept": n_boxes_kept,
            "n_boxes_total_original": n_boxes_total,
            "n_images_now_empty": n_imgs_empty,
        }

    # data.yaml del bucket
    lines = [f"path: {out_root.resolve()}"]
    for split in splits:
        if (dataset_root / "labels" / split).exists():
            lines.append(f"{split}: images/{split}")
    lines.append("")
    lines.append("names:")
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (out_root / "data.yaml").write_text("\n".join(lines), encoding="utf-8")

    return out_root, stats


def main():
    parser = argparse.ArgumentParser(description="Separa un dataset YOLO en 3 subconjuntos por tamano de bbox (small/medium/large).")
    parser.add_argument("--dataset", required=True, type=Path, help="Dataset YOLO existente (con images/, labels/, data.yaml)")
    parser.add_argument("--output-prefix", required=True, type=Path,
                         help="Prefijo de las carpetas de salida: <prefix>_small, <prefix>_medium, <prefix>_large")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "field_work"],
                         help="Splits a procesar (los que no existan se omiten automaticamente)")
    parser.add_argument("--q-low", type=float, default=0.25,
                         help="Cuantil de corte bajo (default 0.25, igual que el paper: XS+S = percentil 0-25)")
    parser.add_argument("--q-high", type=float, default=0.75,
                         help="Cuantil de corte alto (default 0.75, igual que el paper: XL+XXL = percentil 75-100)")
    parser.add_argument("--copy", action="store_true",
                         help="Copiar las imagenes en vez de usar symlinks (usa mas espacio en disco)")
    args = parser.parse_args()

    if not (_HAS_PIL or _HAS_CV2):
        raise SystemExit("Instala Pillow (pip install Pillow) u OpenCV para poder leer el tamano de las imagenes.")

    data_yaml = args.dataset / "data.yaml"
    names = read_data_yaml_names(data_yaml) if data_yaml.exists() else ["object"]

    # --- 1) calcular umbrales SOLO con el split de train (igual que el paper) ---
    print("Calculando umbrales de tamano sobre el split 'train'...")
    train_entries = collect_bbox_areas(args.dataset, "train")
    if not train_entries:
        raise SystemExit("No se encontraron bboxes en train/ -- revisa --dataset.")

    areas = [e[2] for e in train_entries]
    low_thr, high_thr = compute_quantiles(areas, args.q_low, args.q_high)

    print(f"  n bboxes en train: {len(areas)}")
    print(f"  umbral bajo  (percentil {args.q_low*100:.0f}%): {low_thr:.1f} px^2")
    print(f"  umbral alto  (percentil {args.q_high*100:.0f}%): {high_thr:.1f} px^2")
    print(f"  small  = area <= {low_thr:.1f} px^2")
    print(f"  medium = {low_thr:.1f} < area <= {high_thr:.1f} px^2")
    print(f"  large  = area > {high_thr:.1f} px^2")

    # --- 2) generar los 3 datasets ---
    existing_splits = [s for s in args.splits if (args.dataset / "labels" / s).exists()]
    all_stats = {}
    for bucket in ("small", "medium", "large"):
        print(f"\nGenerando dataset '{bucket}'...")
        out_root, stats = write_bucketed_dataset(
            args.dataset, args.output_prefix, bucket, existing_splits,
            low_thr, high_thr, names, use_symlink=not args.copy,
        )
        all_stats[bucket] = stats
        print(f"  -> {out_root}/data.yaml")

    # --- 3) resumen ---
    summary = {
        "source_dataset": str(args.dataset.resolve()),
        "q_low": args.q_low, "q_high": args.q_high,
        "low_threshold_px2": low_thr, "high_threshold_px2": high_thr,
        "n_train_boxes_used_for_thresholds": len(areas),
        "stats_by_bucket": all_stats,
    }
    summary_path = Path(f"{args.output_prefix}_size_buckets_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print("RESUMEN (numero de cajas conservadas por split y bucket)")
    print(f"{'='*70}")
    header = f"{'split':<12}" + "".join(f"{b:>12}" for b in ("small", "medium", "large")) + f"{'total_orig':>12}"
    print(header)
    for split in existing_splits:
        row = f"{split:<12}"
        total_orig = None
        for b in ("small", "medium", "large"):
            s = all_stats[b].get(split, {})
            row += f"{s.get('n_boxes_kept', 0):>12}"
            total_orig = s.get("n_boxes_total_original", total_orig)
        row += f"{total_orig or 0:>12}"
        print(row)

    print(f"\nResumen completo guardado en: {summary_path}")
    print("\nListo. Revisa visualmente con verify_boxes.py, ej.:")
    print(f"  python verify_boxes.py --dataset {args.output_prefix}_small --split train --n 12 --grid")


if __name__ == "__main__":
    main()
