"""Convierte el dataset COCO de polígonos al formato YOLO-seg de segmentación de instancias.

Recorre los mismos splits que prepare_dataset.py, incluido el test de trabajo de campo, y escribe
images/<split>, labels/<split> con una línea por polígono en coordenadas normalizadas y el
data.yaml correspondiente; de las anotaciones con varias partes conserva solo la de mayor área y
descarta las degeneradas. Se apoya en polygon_utils.py y prepara los experimentos de segmentación.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from polygon_utils import polygon_area, normalize_polygon, largest_polygon_part


def find_image_file(images_dir: Path, file_name: str, fallback_ext: str) -> Path | None:
    candidate = images_dir / file_name
    if candidate.exists():
        return candidate
    stem = Path(file_name).stem
    candidate = images_dir / f"{stem}{fallback_ext}"
    if candidate.exists():
        return candidate
    matches = list(images_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    return None


def find_annotation_json(annotations_dir: Path, preferred_stem: str | None = None) -> Path:
    if not annotations_dir.exists():
        print(f"[ERROR] No existe la carpeta de anotaciones: {annotations_dir}")
        sys.exit(1)
    if preferred_stem:
        exact = annotations_dir / f"{preferred_stem}.json"
        if exact.exists():
            return exact
        print(f"[ERROR] No se encontro '{preferred_stem}.json' dentro de {annotations_dir}")
        sys.exit(1)
    matches = sorted(annotations_dir.glob("*.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print(f"[ERROR] No se encontro ningun .json en {annotations_dir}")
        sys.exit(1)
    print(f"[ERROR] Hay varios .json en {annotations_dir}, no se cual usar:")
    for m in matches:
        print(f"    - {m.name}")
    sys.exit(1)


def convert_split(root: Path, split_name: str, output_root: Path, out_split_name: str,
                   img_ext: str, use_symlink: bool, exclude_crowd: bool, global_cat_map: dict | None,
                   min_polygon_points: int = 3):
    json_path = root / split_name / "Annotations" / f"{split_name}_updated.json"
    images_dir = root / split_name / "Images"
    if not json_path.exists():
        print(f"[ERROR] No se encuentra el JSON: {json_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"[ERROR] No se encuentra la carpeta de imagenes: {images_dir}")
        sys.exit(1)
    return _convert_from_json(json_path, images_dir, output_root, out_split_name,
                               img_ext, use_symlink, exclude_crowd, global_cat_map, min_polygon_points)


def convert_field_work(field_work_dir: Path, output_root: Path, img_ext: str, use_symlink: bool,
                        exclude_crowd: bool, global_cat_map: dict | None, json_name_override: str | None,
                        min_polygon_points: int = 3):
    annotations_dir = field_work_dir / "Annotations"
    images_dir = field_work_dir / "Images"
    if not images_dir.exists():
        print(f"[ERROR] No se encuentra la carpeta de imagenes: {images_dir}")
        sys.exit(1)
    json_path = find_annotation_json(annotations_dir, preferred_stem=json_name_override)
    print(f"  json de anotaciones detectado: {json_path.name}")
    return _convert_from_json(json_path, images_dir, output_root, "field_work",
                               img_ext, use_symlink, exclude_crowd, global_cat_map, min_polygon_points)


def _convert_from_json(json_path: Path, images_dir: Path, output_root: Path, out_split_name: str,
                        img_ext: str, use_symlink: bool, exclude_crowd: bool,
                        global_cat_map: dict | None, min_polygon_points: int):
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    out_images = output_root / "images" / out_split_name
    out_labels = output_root / "labels" / out_split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    if global_cat_map is not None and cat_id_to_idx != global_cat_map:
        print(f"[ERROR] Las categorias de '{out_split_name}' no coinciden con las del primer split.")
        sys.exit(1)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    n_imgs_ok, n_imgs_missing, n_polys, n_imgs_no_ann = 0, 0, 0, 0
    n_skipped_degenerate, n_multipart = 0, 0
    poly_areas_px = []

    for img_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        file_name = img_info["file_name"]

        src_img = find_image_file(images_dir, file_name, img_ext)
        if src_img is None:
            n_imgs_missing += 1
            print(f"  [AVISO] Imagen no encontrada en disco: {file_name}")
            continue

        dst_img = out_images / src_img.name
        if not dst_img.exists():
            if use_symlink:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy(src_img, dst_img)

        lines = []
        for ann in anns_by_image.get(img_id, []):
            if exclude_crowd and ann.get("iscrowd", 0) == 1:
                continue

            seg = ann.get("segmentation")
            if not seg:
                continue

            poly, n_parts = largest_polygon_part(seg)
            if n_parts > 1:
                n_multipart += 1

            if len(poly) < min_polygon_points * 2:
                n_skipped_degenerate += 1
                continue

            area_px = polygon_area(poly)
            if area_px <= 0:
                n_skipped_degenerate += 1
                continue

            poly_norm = normalize_polygon(poly, w, h)
            poly_norm = [min(max(v, 0.0), 1.0) for v in poly_norm]  # clamp defensivo

            cls_idx = cat_id_to_idx[ann["category_id"]]
            coords_str = " ".join(f"{v:.6f}" for v in poly_norm)
            lines.append(f"{cls_idx} {coords_str}")

            poly_areas_px.append(area_px)
            n_polys += 1

        if not lines:
            n_imgs_no_ann += 1

        label_path = out_labels / f"{src_img.stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")
        n_imgs_ok += 1

    stats = {
        "split": out_split_name,
        "n_images": n_imgs_ok,
        "n_images_missing": n_imgs_missing,
        "n_images_no_annotation": n_imgs_no_ann,
        "n_polygons": n_polys,
        "n_skipped_degenerate": n_skipped_degenerate,
        "n_multipart_kept_largest": n_multipart,
        "area_px_mean": sum(poly_areas_px) / len(poly_areas_px) if poly_areas_px else 0,
        "area_px_min": min(poly_areas_px) if poly_areas_px else 0,
        "area_px_max": max(poly_areas_px) if poly_areas_px else 0,
    }
    return cat_id_to_idx, categories, stats


def write_data_yaml(output_root: Path, categories: list, split_names: list, has_field_work: bool):
    names_block = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(categories))
    standard = {"train", "val", "test"}
    lines = [f"path: {output_root.resolve()}"]
    for s in split_names:
        lines.append(f"{s}: images/{s}")
    lines.append("")
    lines.append("names:")
    lines.append(names_block)
    if has_field_work:
        lines.append("")
        lines.append("# 'field_work' es un split independiente de validacion externa, separado de test.")
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def print_stats_table(all_stats):
    header = (f"{'split':<12}{'imgs':>7}{'sin_ann':>9}{'polys':>8}"
              f"{'degener.':>10}{'multipart':>11}{'area_px_avg':>13}")
    print(header)
    print("-" * len(header))
    for s in all_stats:
        print(f"{s['split']:<12}{s['n_images']:>7}{s['n_images_no_annotation']:>9}{s['n_polygons']:>8}"
              f"{s['n_skipped_degenerate']:>10}{s['n_multipart_kept_largest']:>11}{s['area_px_mean']:>13.1f}")


def main():
    parser = argparse.ArgumentParser(description="Convierte dataset COCO (poligonos) a formato YOLO-seg.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"])
    parser.add_argument("--img-ext", default=".tif")
    parser.add_argument("--symlink", action="store_true")
    parser.add_argument("--include-crowd", action="store_true")
    parser.add_argument("--min-polygon-points", type=int, default=3,
                         help="Minimo de vertices para conservar un poligono (default: 3, un triangulo)")
    parser.add_argument("--field-work-dir", type=Path, default=None)
    parser.add_argument("--field-work-json-name", default=None)
    args = parser.parse_args()

    exclude_crowd = not args.include_crowd
    args.output.mkdir(parents=True, exist_ok=True)

    global_cat_map, categories_ref, all_stats, out_split_names = None, None, [], []

    for split_name in args.splits:
        out_split_name = split_name.lower()
        out_split_names.append(out_split_name)
        print(f"\n=== Procesando split: {split_name} -> {out_split_name} ===")
        cat_map, categories, stats = convert_split(
            args.root, split_name, args.output, out_split_name, args.img_ext,
            args.symlink, exclude_crowd, global_cat_map, args.min_polygon_points,
        )
        if global_cat_map is None:
            global_cat_map, categories_ref = cat_map, categories
        all_stats.append(stats)

    has_field_work = args.field_work_dir is not None
    if has_field_work:
        print(f"\n=== Procesando Field Work (split aparte) <- {args.field_work_dir} ===")
        cat_map, categories, stats = convert_field_work(
            args.field_work_dir, args.output, args.img_ext, args.symlink,
            exclude_crowd, global_cat_map, args.field_work_json_name, args.min_polygon_points,
        )
        if global_cat_map is None:
            global_cat_map, categories_ref = cat_map, categories
        all_stats.append(stats)
        out_split_names.append("field_work")

    yaml_path = write_data_yaml(args.output, categories_ref, out_split_names, has_field_work)

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Clases: {[(c['id'], c['name']) for c in categories_ref]}")
    print(f"data.yaml generado en: {yaml_path}\n")
    print_stats_table(all_stats)

    total_degenerate = sum(s["n_skipped_degenerate"] for s in all_stats)
    total_multipart = sum(s["n_multipart_kept_largest"] for s in all_stats)
    if total_degenerate:
        print(f"\n[AVISO] {total_degenerate} poligonos descartados por degenerados (menos de "
              f"{args.min_polygon_points} vertices o area 0).")
    if total_multipart:
        print(f"[AVISO] {total_multipart} anotaciones tenian mas de una parte de poligono "
              f"(objeto partido) -- se conservo solo la parte de mayor area en cada una.")

    print("\nListo. Revisa visualmente con verify_polygons.py antes de entrenar.")


if __name__ == "__main__":
    main()
