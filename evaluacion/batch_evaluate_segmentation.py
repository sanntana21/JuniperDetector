"""Evalúa en lote las máscaras de segmentación crudas de los dos conjuntos de test.

Lee las predicciones en RLE del test fotointerpretado y del test de trabajo de campo, las
convierte a polígonos y aplica supresión de no máximos para que un mismo arbusto no cuente
varias veces bajo S-IoU. Mide IoU y S-IoU a confianza fija y calibrada, y guarda
results/_batch_segmentation/all_results_segm.json.
"""
import json
import glob
import re
import os
import argparse
from collections import defaultdict

from pycocotools import mask as mask_utils
from skimage import measure
from scipy.ndimage import binary_dilation
import numpy as np

from segmentation_analisis import run_seg_analysis
from analize_results import compute_precision_recall


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1


def rle_to_polygon(rle_seg, dilate=True, dilation_iters=3):
    """Igual que process_results.py: decodifica RLE, dilata opcionalmente,
    y se queda con el contorno mas largo como poligono (lista plana
    x1,y1,x2,y2,...)."""
    binary_mask = mask_utils.decode(rle_seg).astype(bool)
    if dilate and dilation_iters > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilation_iters)
    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
    if not contours:
        return []
    largest = max(contours, key=len)
    largest = np.fliplr(largest)  # (row,col) -> (x,y)
    return [largest.flatten().tolist()]


def polygon_iou(poly_a, poly_b):
    """IoU real entre dos poligonos shapely (no cajas)."""
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.area == 0 or poly_b.area == 0:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.area + poly_b.area - inter
    return inter / union if union > 0 else 0.0


def polygon_nms(anns, iou_thr=0.7):
    """NMS greedy sobre poligonos (no cajas), por score, dentro de una
    misma imagen. Ausente por completo en el process_results.py original
    para el flujo de segmentacion -- causa raiz confirmada de que un mismo
    objeto puede terminar contando como decenas de TP independientes bajo
    S-IoU (Ec. 2 evalua cada prediccion por separado, sin limite de que
    TP <= numero de objetos reales), inflando artificialmente Precision,
    Recall y F1 sin que el modelo haya detectado nada mas.

    iou_thr mas bajo que el 0.9 usado en bbox porque los poligonos de
    detecciones "duplicadas" reales (el mismo objeto repetido por el
    modelo) suelen solaparse con menor IoU que dos cajas identicas,
    debido al ruido normal de la reconstruccion RLE->contorno. Ajustable.
    """
    from shapely.geometry import Polygon

    order = sorted(range(len(anns)), key=lambda i: -anns[i]["score"])
    polys = []
    for a in anns:
        coords = a["segmentation"][0]
        pts = list(zip(coords[0::2], coords[1::2]))
        polys.append(Polygon(pts) if len(pts) >= 3 else None)

    keep = []
    suppressed = set()
    for idx in order:
        if idx in suppressed or polys[idx] is None:
            continue
        keep.append(idx)
        for j in order:
            if j == idx or j in suppressed or polys[j] is None:
                continue
            if polygon_iou(polys[idx], polys[j]) >= iou_thr:
                suppressed.add(j)
    return [anns[i] for i in keep]


def process_raw_segm(raw_path, images_path, out_path, dilation_iters=3,
                      img_ext_glob="*.tif*", nms_iou_thr=0.7):
    """Reimplementacion en Python puro del branch de segmentacion de
    process_results.py: remapea image_id (indice interno del detector ->
    numero real de archivo), convierte cada mascara RLE a poligono, Y
    ADEMAS aplica NMS por imagen (poligono a poligono) antes de guardar --
    el original NUNCA hacia esto para segmentacion (solo para bbox),
    permitiendo que un mismo objeto contara como decenas de TP
    independientes bajo S-IoU. Ver polygon_nms() para el detalle.
    """
    files = glob.glob(f"{images_path}/{img_ext_glob}")
    lista_jpg = {i: extract_number(j.split("/")[-1])
                 for i, j in enumerate(sorted(files, key=extract_number))}

    with open(raw_path) as f:
        coco = json.load(f)

    new_json = []
    skipped_empty = 0
    for entry in coco:
        internal_id = entry["image_id"]
        if internal_id not in lista_jpg:
            continue
        real_id = lista_jpg[internal_id]
        poly = rle_to_polygon(entry["segmentation"], dilate=True, dilation_iters=dilation_iters)
        if not poly:
            skipped_empty += 1
            continue
        new_json.append({
            "image_id": int(real_id),
            "segmentation": poly,
            "score": entry["score"],
            "category": 1,
        })

    grouped = defaultdict(list)
    for ann in new_json:
        grouped[ann["image_id"]].append(ann)

    filtered = []
    n_before_nms = len(new_json)
    for image_id, anns in grouped.items():
        filtered.extend(polygon_nms(anns, iou_thr=nms_iou_thr))

    with open(out_path, "w") as f:
        json.dump(filtered, f)

    return len(coco), n_before_nms, len(filtered), skipped_empty


def evaluate_one(tag, mode, raw_path, images_path, tmp_dir, fixed_confidence,
                  metric_thr=0.5, dilation_iters=3, nms_iou_thr=0.7):
    os.makedirs(tmp_dir, exist_ok=True)
    processed_path = f"{tmp_dir}/{tag}_processed_segm.json"
    n_raw, n_before_nms, n_after_nms, n_empty = process_raw_segm(
        raw_path, images_path, processed_path, dilation_iters=dilation_iters, nms_iou_thr=nms_iou_thr
    )
    print(f"    {tag}: {n_raw} mascaras crudas -> {n_before_nms} poligonos validos "
          f"-> {n_after_nms} tras NMS (iou_thr={nms_iou_thr}) [{n_empty} vacios/descartados]")

    result = {"tag": tag, "n_raw": n_raw, "n_before_nms": n_before_nms,
              "n_after_nms": n_after_nms, "n_empty": n_empty}

    for metric in ("iou", "siou"):
        F1, TP, FP, FN = run_seg_analysis(
            mode=mode, confidence_threshold=fixed_confidence, metric_thr=metric_thr,
            metric=metric, results_file_override=processed_path, dilation_iters=dilation_iters,
        )
        p, r = compute_precision_recall(TP, FP, FN)
        result[f"{metric}_fixed"] = {"confidence": fixed_confidence, "TP": TP, "FP": FP, "FN": FN,
                                      "precision": p, "recall": r, "F1": F1}

        best = None
        for conf in [i / 20 for i in range(0, 21)]:
            F1b, TPb, FPb, FNb = run_seg_analysis(
                mode=mode, confidence_threshold=conf, metric_thr=metric_thr,
                metric=metric, results_file_override=processed_path, dilation_iters=dilation_iters,
            )
            if best is None or F1b > best["F1"]:
                pb, rb = compute_precision_recall(TPb, FPb, FNb)
                best = {"confidence": conf, "TP": TPb, "FP": FPb, "FN": FNb,
                        "precision": pb, "recall": rb, "F1": F1b}
        result[f"{metric}_best"] = best

    return result


def print_paper_style_table(results, title, fixed_confidence):
    print(f"\n{'='*100}")
    print(f"  {title}  (SEGMENTACION -- confianza fija = {fixed_confidence}, umbral overlap = 50%)")
    print(f"{'='*100}")
    header = f"{'Modelo':<28}{'':>4}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>8}{'Rec':>8}{'F1':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        for metric, label in (("iou_fixed", "IoU*"), ("siou_fixed", "S-IoU*"),
                               ("iou_best", "IoU best"), ("siou_best", "S-IoU best")):
            m = r[metric]
            name = r["tag"] if metric == "iou_fixed" else ""
            print(f"{name:<28}{label:>10}{m['TP']:>6}{m['FP']:>6}{m['FN']:>6}"
                  f"{m['precision']*100:>7.2f}%{m['recall']*100:>7.2f}%{m['F1']*100:>7.2f}%")
        print("-" * len(header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dilation", type=int, default=3,
                         help="Iteraciones de dilatacion al pasar RLE a poligono (default: 3, igual que process_results.py)")
    parser.add_argument("--nms-iou", type=float, default=0.7,
                         help="Umbral IoU para el NMS de poligonos antes de evaluar (default: 0.7). "
                              "Ausente por completo en el process_results.py original para "
                              "segmentacion -- sin esto, un mismo objeto repetido por el modelo "
                              "puede contar como muchos TP independientes bajo S-IoU, inflando "
                              "Precision/Recall/F1 artificialmente.")
    args = parser.parse_args()

    os.makedirs("results/_batch_segmentation", exist_ok=True)

    # ---------------- TEST / PI ----------------
    test_dir = "Photo_Interpretation_Data/Test"
    test_images = f"{test_dir}/Images"
    test_models = {
        "co_dino": f"{test_dir}/co_dino_output.segm.json",
        "swimt": f"{test_dir}/swimt_output.segm.json",
        # Anade aqui mas modelos si tienes mas *.segm.json crudos, ej.:
        # "mask2former": f"{test_dir}/mask2former_output.segm.json",
    }

    test_results = []
    for tag, path in test_models.items():
        if not os.path.exists(path):
            print(f"[AVISO] no existe: {path}")
            continue
        print(f"Procesando TEST/{tag} (segmentacion) ...")
        r = evaluate_one(tag, mode=0, raw_path=path, images_path=test_images,
                          tmp_dir="results/_batch_segmentation/test_tmp",
                          fixed_confidence=0.9, metric_thr=0.5, dilation_iters=args.dilation,
                          nms_iou_thr=args.nms_iou)
        test_results.append(r)

    print_paper_style_table(test_results, "PHOTO-INTERPRETED (PI) TEST SET", fixed_confidence=0.9)

    # ---------------- FIELD WORK ----------------
    fw_dir = "Field_Work_Data/External_Val_Data"
    fw_images = f"{fw_dir}/Images"
    fw_models = {
        "swimt": f"{fw_dir}/swimt_output.segm.json",
        # co_dino no tiene salida de segmentacion cruda para Field Work en
        # este paquete -- si la generas, anadela aqui.
    }

    fw_results = []
    for tag, path in fw_models.items():
        if not os.path.exists(path):
            print(f"[AVISO] no existe: {path}")
            continue
        print(f"Procesando FW/{tag} (segmentacion) ...")
        r = evaluate_one(tag, mode=1, raw_path=path, images_path=fw_images,
                          tmp_dir="results/_batch_segmentation/fw_tmp",
                          fixed_confidence=0.5, metric_thr=0.5, dilation_iters=args.dilation,
                          nms_iou_thr=args.nms_iou)
        fw_results.append(r)

    print_paper_style_table(fw_results, "FIELD WORK (FW) TEST SET", fixed_confidence=0.5)

    with open("results/_batch_segmentation/all_results_segm.json", "w") as f:
        json.dump({"test": test_results, "field_work": fw_results}, f, indent=2)
    print("\nResultados completos guardados en: results/_batch_segmentation/all_results_segm.json")
