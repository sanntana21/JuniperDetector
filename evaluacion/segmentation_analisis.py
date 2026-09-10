"""Evalúa las predicciones de segmentación barriendo el umbral de confianza con IoU y S-IoU.

Lee las anotaciones COCO de referencia y las predicciones procesadas del test
fotointerpretado o del test de trabajo de campo, convirtiendo máscaras RLE y polígonos a
geometría shapely. Guarda en results/<etiqueta>/ el barrido completo en sweep_*.json, la
curva de F1 frente a la confianza en PNG y las tablas resumen en summary.json y
summary_*.txt.
"""

import json
import os
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pycocotools import mask as mask_utils
from skimage import measure
import matplotlib.pyplot as plt
import argparse

from utils import extract_number
# Reutilizamos las funciones de evaluación existentes
from analize_results import siou_evaluation, match_greedy_iou, iou_matrix, compute_precision_recall


def rle_to_shapely(rle_seg, dilation_iters=0):
    """Convierte RLE a polígono Shapely (el mayor contorno)."""
    from scipy.ndimage import binary_dilation

    binary_mask = mask_utils.decode(rle_seg).astype(bool)
    if dilation_iters > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilation_iters)

    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
    if not contours:
        return None

    largest = max(contours, key=len)
    largest = np.fliplr(largest)  # (row,col) -> (x,y)

    if len(largest) < 3:
        return None

    poly = Polygon(largest)
    return poly if poly.is_valid else poly.buffer(0)


def polygon_list_to_shapely(seg_list):
    """Convierte segmentación COCO polygon format a Shapely."""
    polys = []
    for seg in seg_list:
        coords = list(zip(seg[::2], seg[1::2]))
        if len(coords) >= 3:
            p = Polygon(coords)
            polys.append(p if p.is_valid else p.buffer(0))
    if not polys:
        return None
    return unary_union(polys)


def ann_to_shapely(ann):
    """Detecta el formato de segmentación y devuelve un polígono Shapely."""
    seg = ann.get("segmentation")
    if seg is None:
        return None

    if isinstance(seg, dict):  # RLE (predicciones)
        return rle_to_shapely(seg)
    elif isinstance(seg, list):  # Polygon (ground truth COCO)
        return polygon_list_to_shapely(seg)

    return None


def run_seg_analysis(
    mode=0,
    confidence_threshold=0.5,
    metric_thr=0.5,
    metric="iou",  # "iou" or "siou"
    verbose=0,
    dilation_iters=None,
    results_file_override=None,
):
    extension = ""
    segmentation = True
    output_type = "segm" if segmentation else "bbox"
    # BUG ARREGLADO: aqui habia un 'mode = 1' que sobrescribia SIEMPRE el
    # parametro recibido, forzando a analizar Field Work sin importar que
    # se pidiera Test/PI. Se quita para respetar el 'mode' pasado.
    if mode == 1:
        data_dir = "Field_Work_Data/External_Val_Data"
        results_path = results_file_override or f"{data_dir}/process_results_{output_type}.json"
        original_path = f"{data_dir}/Annotations/FieldWork_updated.json"
    else:
        data_dir = "Photo_Interpretation_Data/Test"
        results_path = results_file_override or f"{data_dir}/process_results_{output_type}.json"
        original_path = f"{data_dir}/Annotations/Test_updated.json"

    with open(original_path) as f:
        coco_original = json.load(f)

    with open(results_path) as f:
        coco_results = json.load(f)

    coco_results = [a for a in coco_results if a.get("score", 0) >= confidence_threshold]

    # BUG ARREGLADO: 'coco_original["images"][ann["image_id"]]' indexaba la
    # lista de imagenes POR POSICION usando el image_id de la anotacion.
    # Eso solo es correcto si el campo 'id' de cada imagen coincide con su
    # posicion en la lista -- se cumple en Test_updated.json (por
    # casualidad, no por diseno) pero NO en FieldWork_updated.json, donde
    # 'id' es un string ordenado alfabeticamente ('0','1','10','100'...) y
    # no coincide con la posicion en el 124/124 de los casos. Ademas, al
    # ser un string, ni siquiera es un indice de lista valido -> crash.
    #
    # Arreglo correcto: construir un diccionario id -> info_imagen, y
    # buscar por el VALOR real de id (normalizado a string, ya que el json
    # mezcla int en Test y str en FieldWork), no por posicion.
    images_by_id = {str(im["id"]): im for im in coco_original["images"]}

    gt_anns = [
        {
            **ann,
            "image_id": extract_number(
                images_by_id[str(ann["image_id"])]["file_name"]
            )
        }
        for ann in coco_original["annotations"]
    ]

    for b in gt_anns:
        b["image_id"] = int(b["image_id"])
    image_ids = sorted(set(a["image_id"] for a in gt_anns))

    TP_total = FP_total = FN_total = 0
    

    for img_id in image_ids:
        gt_img_anns = [a for a in gt_anns if a["image_id"] == img_id]
        pred_img_anns = [a for a in coco_results if a["image_id"] == img_id]
        
        gt_polys_raw = [ann_to_shapely(a) for a in gt_img_anns]
        pred_polys_raw = [ann_to_shapely(a) for a in pred_img_anns]

        gt_polys = [p for p in gt_polys_raw if p is not None]
        pred_polys = [p for p in pred_polys_raw if p is not None]
        
        if verbose:
            print(f"[img {img_id}] GT raw: {len(gt_polys_raw)} → válidos: {len(gt_polys)} | "f"Pred raw: {len(pred_polys_raw)} → válidos: {len(pred_polys)}")

        if metric == "siou":
            TP, FP, FN = siou_evaluation(pred_polys, gt_polys, metric_thr)

        else:  # iou estándar sobre polígonos
            if gt_polys and pred_polys:
                # Matriz IoU usando área de intersección/unión de polígonos
                M = np.zeros((len(pred_polys), len(gt_polys)))
                for i, p in enumerate(pred_polys):
                    for j, g in enumerate(gt_polys):
                        inter = p.intersection(g).area
                        union = p.union(g).area
                        M[i, j] = inter / union if union > 0 else 0.0

                matched_pred, matched_gt = match_greedy_iou(M, metric_thr)
                TP = len(matched_pred)
                FP = len(pred_polys) - TP
                FN = len(gt_polys) - TP
            else:
                TP = 0
                FP = len(pred_polys)
                FN = len(gt_polys)

        TP_total += TP
        FP_total += FP
        FN_total += FN
    
    print("\n____________",
    confidence_threshold,
    metric_thr,
    metric,"_______\n")
    print("TP",TP_total,"FP",FP_total,"FN", FN_total)

    precision, recall = compute_precision_recall(TP_total, FP_total, FN_total)
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return F1, TP_total, FP_total, FN_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, choices=[0, 1], default=0)
    parser.add_argument("--dilation", type=int, default=2, help="Iteraciones de dilatación para RLE")
    parser.add_argument(
        "--tag",
        default=None,
        help="Etiqueta para identificar esta corrida (ej. 'mask2former', 'swimt_seg'). "
             "Los resultados se guardan en results/<tag>/, evitando que corridas de "
             "distintos modelos se sobrescriban entre si. Default: 'run'.",
    )
    args = parser.parse_args()

    mode = args.mode
    tag = args.tag or "run"
    title = "PI" if mode == 0 else "FW"
    fixed_confidence = 0.9 if mode == 0 else 0.5
    confidence_values = [i / 10 for i in range(0, 11)]

    out_dir = f"results/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Guardando resultados en: {out_dir}/")

    full_summary = {"tag": tag, "mode": mode, "title": title, "by_threshold": {}}

    for metric_thr in [0.5, 0.75]:
        results_iou = []
        results_siou = []

        for confidence in confidence_values:
            F1, TP, FP, FN = run_seg_analysis(
                mode=mode, confidence_threshold=confidence,
                metric_thr=metric_thr, metric="iou",
                dilation_iters=args.dilation
            )
            precision, recall = compute_precision_recall(TP, FP, FN)
            results_iou.append({"confidence": confidence, "F1": F1, "TP": TP,
                                 "FP": FP, "FN": FN, "precision": precision, "recall": recall})

        for confidence in confidence_values:
            F1, TP, FP, FN = run_seg_analysis(
                mode=mode, confidence_threshold=confidence,
                metric_thr=metric_thr, metric="siou",
                dilation_iters=args.dilation
            )
            precision, recall = compute_precision_recall(TP, FP, FN)
            results_siou.append({"confidence": confidence, "F1": F1, "TP": TP,
                                  "FP": FP, "FN": FN, "precision": precision, "recall": recall})

        # ---------- Guardar el barrido completo (antes se perdia) ----------
        sweep_path = f"{out_dir}/sweep_{title}_SEG_thr{metric_thr}.json"
        with open(sweep_path, "w") as f:
            json.dump({"tag": tag, "mode": mode, "title": title, "metric_thr": metric_thr,
                       "iou": results_iou, "siou": results_siou}, f, indent=2)

        # Gráfica
        image = f"{out_dir}/{title}_SEG_thr{metric_thr}"
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_values, [r["F1"] for r in results_iou],
                 color='green', marker='o', label='IoU (seg)', linewidth=2)
        plt.plot(confidence_values, [r["F1"] for r in results_siou],
                 color='gold', marker='s', label='SIoU (seg)', linewidth=2)
        plt.xlabel("Confidence threshold")
        plt.ylabel("F1 score")
        plt.title(f"{tag} — {title} Segmentation — Metric Threshold {metric_thr}")
        plt.legend()
        plt.grid(True)
        plt.savefig(image + ".png", dpi=300, bbox_inches="tight")
        plt.close()

        # Tabla resumen
        best_iou = max(results_iou, key=lambda x: x["F1"])
        best_siou = max(results_siou, key=lambda x: x["F1"])
        fixed_iou = next(r for r in results_iou if r["confidence"] == fixed_confidence)
        fixed_siou = next(r for r in results_siou if r["confidence"] == fixed_confidence)

        full_summary["by_threshold"][str(metric_thr)] = {
            "fixed_confidence": fixed_confidence,
            "IoU_fixed": fixed_iou, "SIoU_fixed": fixed_siou,
            "IoU_best": best_iou, "SIoU_best": best_siou,
            "sweep_file": sweep_path, "plot": image + ".png",
        }

        lines = []
        lines.append(f"\n{'='*90}")
        lines.append(f"SEGMENTACIÓN ({tag} | {title} — threshold {metric_thr})")
        lines.append(f"{'='*90}")
        lines.append(f"{'Metric':<8} | {'Conf':<5} | {'TP':<5} | {'FP':<5} | {'FN':<5} | "
                      f"{'Precision':<9} | {'Recall':<6} | {'F1':<6}")
        lines.append("-" * 90)

        def row(metric, r):
            return (f"{metric:<8} | {r['confidence']:<5.2f} | {r['TP']:<5} | {r['FP']:<5} | "
                    f"{r['FN']:<5} | {r['precision']*100:<9.3f} | {r['recall']*100:<6.3f} | {r['F1']*100:<6.3f}")

        lines.append(row("IoU*", fixed_iou))
        lines.append(row("SIoU*", fixed_siou))
        lines.append("-" * 90)
        lines.append(row("IoU best", best_iou))
        lines.append(row("SIoU best", best_siou))
        lines.append("=" * 90)
        lines.append("* = umbral de confianza fijo de referencia; sin * = mejor F1 del barrido completo")

        table_text = "\n".join(lines)
        print(table_text)
        with open(f"{out_dir}/summary_{title}_SEG_thr{metric_thr}.txt", "w") as f:
            f.write(table_text + "\n")

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\nTodo guardado en: {out_dir}/")