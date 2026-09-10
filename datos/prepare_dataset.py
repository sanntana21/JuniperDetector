"""Convierte el dataset COCO de detección al formato YOLO que espera Ultralytics.

Lee los JSON de anotaciones y las imágenes de los splits Train, Val y Test y, si se indica, los del
test de trabajo de campo, y escribe images/<split>, labels/<split> con una caja normalizada por
línea, el data.yaml del dataset y un resumen por consola de imágenes y anotaciones convertidas. El
test de trabajo de campo se mantiene como split independiente, field_work, que no se mezcla con test.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def find_image_file(images_dir: Path, file_name: str, fallback_ext: str) -> Path | None:
    """Localiza el archivo de imagen real en disco, siendo tolerante a
    discrepancias de extension entre el json y los archivos reales."""
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


def convert_split(
    root: Path,
    split_name: str,
    output_root: Path,
    out_split_name: str,
    img_ext: str,
    use_symlink: bool,
    exclude_crowd: bool,
    global_cat_map: dict | None,
):
    """Convierte un split principal (Train/Val/Test), asumiendo el patron de
    nombre fijo '{split_name}_updated.json' dentro de Annotations/."""
    json_path = root / split_name / "Annotations" / f"{split_name}_updated.json"
    images_dir = root / split_name / "Images"

    if not json_path.exists():
        print(f"[ERROR] No se encuentra el JSON: {json_path}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"[ERROR] No se encuentra la carpeta de imagenes: {images_dir}")
        sys.exit(1)

    return _convert_from_json(
        json_path=json_path,
        images_dir=images_dir,
        output_root=output_root,
        out_split_name=out_split_name,
        img_ext=img_ext,
        use_symlink=use_symlink,
        exclude_crowd=exclude_crowd,
        global_cat_map=global_cat_map,
    )


def find_annotation_json(annotations_dir: Path, preferred_stem: str | None = None) -> Path:
    """Localiza el json de anotaciones dentro de Annotations/ SIN asumir un
    patron de nombre fijo (usado solo para Field Work, cuyo json no se llama
    '{split}_updated.json'). Si hay un unico .json en la carpeta, se usa
    directamente; si hay varios, se requiere 'preferred_stem' para elegir."""
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
    print("Desambigua con --field-work-json-name <nombre_sin_extension>")
    sys.exit(1)


def convert_field_work(
    field_work_dir: Path,
    output_root: Path,
    img_ext: str,
    use_symlink: bool,
    exclude_crowd: bool,
    global_cat_map: dict | None,
    json_name_override: str | None,
):
    """Convierte el dataset de Field Work (validacion externa), tratado
    como split independiente 'field_work' (nunca se mezcla con test)."""
    annotations_dir = field_work_dir / "Annotations"
    images_dir = field_work_dir / "Images"

    if not images_dir.exists():
        print(f"[ERROR] No se encuentra la carpeta de imagenes: {images_dir}")
        sys.exit(1)

    json_path = find_annotation_json(annotations_dir, preferred_stem=json_name_override)
    print(f"  json de anotaciones detectado: {json_path.name}")

    return _convert_from_json(
        json_path=json_path,
        images_dir=images_dir,
        output_root=output_root,
        out_split_name="field_work",
        img_ext=img_ext,
        use_symlink=use_symlink,
        exclude_crowd=exclude_crowd,
        global_cat_map=global_cat_map,
    )


def _convert_from_json(
    json_path: Path,
    images_dir: Path,
    output_root: Path,
    out_split_name: str,
    img_ext: str,
    use_symlink: bool,
    exclude_crowd: bool,
    global_cat_map: dict | None,
):
    """Logica de conversion COCO->YOLO compartida por splits principales y
    por Field Work. No se expone directamente: se llama desde convert_split
    y convert_field_work."""
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
        print(f"        {out_split_name}: {cat_id_to_idx}")
        print(f"        esperado: {global_cat_map}")
        sys.exit(1)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    n_imgs_ok, n_imgs_missing, n_anns, n_anns_crowd_skipped = 0, 0, 0, 0
    n_imgs_no_ann = 0
    bbox_widths, bbox_heights = [], []

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
        anns = anns_by_image.get(img_id, [])
        for ann in anns:
            if exclude_crowd and ann.get("iscrowd", 0) == 1:
                n_anns_crowd_skipped += 1
                continue

            x_min, y_min, bw, bh = ann["bbox"]

            if bw <= 0 or bh <= 0:
                print(f"  [AVISO] bbox degenerada en {file_name} (ann id {ann.get('id')}): {ann['bbox']}")
                continue

            x_center = (x_min + bw / 2) / w
            y_center = (y_min + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            bw_n = min(max(bw_n, 0.0), 1.0)
            bh_n = min(max(bh_n, 0.0), 1.0)

            cls_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}")

            bbox_widths.append(bw)
            bbox_heights.append(bh)
            n_anns += 1

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
        "n_annotations": n_anns,
        "n_crowd_skipped": n_anns_crowd_skipped,
        "bbox_w_mean": sum(bbox_widths) / len(bbox_widths) if bbox_widths else 0,
        "bbox_h_mean": sum(bbox_heights) / len(bbox_heights) if bbox_heights else 0,
        "bbox_w_min": min(bbox_widths) if bbox_widths else 0,
        "bbox_w_max": max(bbox_widths) if bbox_widths else 0,
    }
    return cat_id_to_idx, categories, stats


def write_data_yaml(output_root: Path, categories: list, split_names: list, has_field_work: bool):
    names_block = "\n".join(f"  {i}: {c['name']}" for i, c in enumerate(categories))
    split_map = {"train": "images/train", "val": "images/val", "test": "images/test"}
    lines = [f"path: {output_root.resolve()}"]
    for s in split_names:
        if s in split_map:
            lines.append(f"{s}: {split_map[s]}")
    if has_field_work:
        lines.append("field_work: images/field_work")
    lines.append("")
    lines.append("names:")
    lines.append(names_block)

    if has_field_work:
        lines.append("")
        lines.append("# 'field_work' es un split independiente de validacion externa, NO forma")
        lines.append("# parte de test. Usa --split field_work en verify_boxes.py / train_btm.py")
        lines.append("# para acceder a el explicitamente.")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def print_stats_table(all_stats):
    header = f"{'split':<12}{'imgs':>7}{'sin_ann':>9}{'faltan':>8}{'anns':>8}{'bbox_w_avg':>12}{'bbox_h_avg':>12}"
    print(header)
    print("-" * len(header))
    for s in all_stats:
        print(
            f"{s['split']:<12}{s['n_images']:>7}{s['n_images_no_annotation']:>9}"
            f"{s['n_images_missing']:>8}{s['n_annotations']:>8}"
            f"{s['bbox_w_mean']:>12.1f}{s['bbox_h_mean']:>12.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Convierte dataset COCO (Train/Val/Test [+ Field Work]) a formato YOLO.")
    parser.add_argument("--root", required=True, type=Path, help="Carpeta raiz que contiene Train/Val/Test")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta de salida para el dataset YOLO")
    parser.add_argument("--splits", nargs="+", default=["Train", "Val", "Test"],
                         help="Nombres exactos de las subcarpetas de split (por defecto: Train Val Test)")
    parser.add_argument("--img-ext", default=".tif", help="Extension de imagen esperada (default: .tif)")
    parser.add_argument("--symlink", action="store_true",
                         help="Usar symlinks en vez de copiar las imagenes (ahorra espacio en disco)")
    parser.add_argument("--include-crowd", action="store_true",
                         help="Incluir anotaciones marcadas como iscrowd=1 (por defecto se excluyen)")
    parser.add_argument("--field-work-dir", type=Path, default=None,
                         help="Ruta a la carpeta de Field Work data (contiene Annotations/ e Images/). "
                              "Se procesa como split independiente 'field_work', NUNCA se mezcla con test.")
    parser.add_argument("--field-work-json-name", default=None,
                         help="Nombre exacto (sin .json) del archivo de anotaciones de Field Work, "
                              "solo necesario si hay mas de un .json en su carpeta Annotations/.")
    args = parser.parse_args()

    exclude_crowd = not args.include_crowd
    args.output.mkdir(parents=True, exist_ok=True)

    global_cat_map = None
    categories_ref = None
    all_stats = []
    out_split_names = []

    for split_name in args.splits:
        out_split_name = split_name.lower()
        out_split_names.append(out_split_name)
        print(f"\n=== Procesando split: {split_name} -> {out_split_name} ===")

        cat_map, categories, stats = convert_split(
            root=args.root,
            split_name=split_name,
            output_root=args.output,
            out_split_name=out_split_name,
            img_ext=args.img_ext,
            use_symlink=args.symlink,
            exclude_crowd=exclude_crowd,
            global_cat_map=global_cat_map,
        )

        if global_cat_map is None:
            global_cat_map = cat_map
            categories_ref = categories

        all_stats.append(stats)

    has_field_work = args.field_work_dir is not None
    if has_field_work:
        print(f"\n=== Procesando Field Work data (split aparte, no es test) <- {args.field_work_dir} ===")
        cat_map, categories, stats = convert_field_work(
            field_work_dir=args.field_work_dir,
            output_root=args.output,
            img_ext=args.img_ext,
            use_symlink=args.symlink,
            exclude_crowd=exclude_crowd,
            global_cat_map=global_cat_map,
            json_name_override=args.field_work_json_name,
        )
        if global_cat_map is None:
            global_cat_map = cat_map
            categories_ref = categories
        all_stats.append(stats)

    yaml_path = write_data_yaml(args.output, categories_ref, out_split_names, has_field_work)

    # --- resumen final ---
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Clases detectadas: {[(c['id'], c['name']) for c in categories_ref]}")
    print(f"data.yaml generado en: {yaml_path}\n")
    print_stats_table(all_stats)

    total_missing = sum(s["n_images_missing"] for s in all_stats)
    total_no_ann = sum(s["n_images_no_annotation"] for s in all_stats)
    if total_missing > 0:
        print(f"\n[AVISO] {total_missing} imagenes referenciadas en el JSON no se encontraron en disco.")
    if total_no_ann > 0:
        print(f"[AVISO] {total_no_ann} imagenes quedaron sin ninguna anotacion (negativos). "
              f"Revisa si es esperado o si hay que descartarlas.")
    if has_field_work:
        print(f"\n[INFO] Field Work procesado como split 'field_work', independiente de test.")

    print("\nListo. Antes de entrenar, revisa visualmente algunas cajas convertidas")
    print("(usa verify_boxes.py --split <nombre>) para confirmar que la conversion es correcta.")


if __name__ == "__main__":
    main()