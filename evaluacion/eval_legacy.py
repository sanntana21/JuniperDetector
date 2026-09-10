#!/usr/bin/env python3
"""Evalúa con IoU y S-IoU las arquitecturas transformer a partir de sus detecciones crudas.

Recorre las salidas de Co-DETR, Co-DINO, Co-Deformable-DETR y Mask2Former sobre el test
fotointerpretado y el test de trabajo de campo, barriendo el umbral de confianza y dos
umbrales de solape. Escribe una fila por combinación en analisis/legacy_modelos.csv, que
alimenta las tablas comparativas del trabajo.
"""
import json, sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana")
sys.path.insert(0, str(SRV / "model_soup" / "scripts"))
from siou_metrics_polygon import evaluate_dataset      # noqa: E402

# --- aceleracion: memoizacion de la construccion de poligonos shapely ---
import functools
import siou_metrics_polygon as _smp

_orig_to_shapely = _smp._to_shapely


@functools.lru_cache(maxsize=200_000)
def _cached(coords_tuple):
    return _orig_to_shapely(list(coords_tuple))


def _to_shapely_cached(coords_flat):
    return _cached(tuple(coords_flat))


_smp._to_shapely = _to_shapely_cached
# ------------------------------------------------------------------------


ANN = SRV / "Co-DETR" / "data" / "juniper" / "annotations"
GT_FILES = {"test": ANN / "instances_test2017.json",
            "field_work": ANN / "field_work_data.json"}

# modelo -> (etiqueta, {split: fichero de detecciones}, tipo de geometria)
MODELOS = {
    "codetr_r50":        ("Co-DETR ResNet-50", {
        "test":       "exp_tiff/test_results/codetr_output.bbox.json",
        "field_work": "exp_tiff/field_work_results/codetr_output.bbox.json"}, "bbox"),
    "codetr_r50_aug":    ("Co-DETR R50 + aug/LR ciclico", {
        "test":       "exp_tiff_data_augmentation_cicled_lr/test_results/codetr_output.bbox.json"}, "bbox"),
    "codetr_r50_v1":     ("Co-DETR R50 (config inicial)", {
        "test":       "exp_v1/test_results/codetr_output.bbox.json"}, "bbox"),
    "co_deform_swinL":   ("Co-Deformable-DETR Swin-L", {
        "test":       "exp_co_deformable_detr_swimL_juniper/test_results/co_dino_output.bbox.json"}, "bbox"),
    "co_dino_swinL":     ("Co-DINO Swin-L", {
        "test":       "exp_co_dino_swimL_juniper/test_results/co_dino_output.bbox.json",
        "field_work": "exp_co_dino_swimL_juniper/fw_results/co_dino_output.bbox.json"}, "bbox"),
    "co_dino_swinL_v1":  ("Co-DINO Swin-L (v1)", {
        "test":       "exp_co_dino_swimL_juniper_v1/test_results/codetr_output.bbox.json",
        "field_work": "exp_co_dino_swimL_juniper_v1/fw_results/co_dino_output.bbox.json"}, "bbox"),
    "mask2former_swinT": ("Mask2Former Swin-T", {
        "test":       "experimento_mask2former/test_results/mask2former_output.bbox.json",
        "field_work": "experimento_mask2former/field_work_results/swimt_output.bbox.json"}, "bbox"),
    "mask2former_cos":   ("Mask2Former Swin-T (coseno)", {
        "test":       "experimento_mask2former_cossine/test_results/codetr_output.bbox.json"}, "bbox"),
}

THRESHOLDS = [0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50,
              0.60, 0.70, 0.80, 0.90]


def bbox_to_poly(b):
    x, y, w, h = b
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def load_gt(split):
    d = json.loads(GT_FILES[split].read_text())
    per_img = {im["id"]: [] for im in d["images"]}
    for a in d["annotations"]:
        per_img[a["image_id"]].append(bbox_to_poly(a["bbox"]))
    return per_img


def job(args):
    key, label, split, relpath, geom = args
    f = SRV / "Co-DETR" / relpath
    if not f.exists():
        return []
    gt = load_gt(split)
    dets = json.loads(f.read_text())
    per_pred = {i: ([], []) for i in gt}
    for d in dets:
        i = d["image_id"]
        if i not in per_pred:
            continue
        per_pred[i][0].append(bbox_to_poly(d["bbox"]))
        per_pred[i][1].append(d["score"])
    ids = sorted(gt)
    preds = [per_pred[i] for i in ids]
    gts = [gt[i] for i in ids]
    out = []
    for ov in (0.5, 0.75):
        for st in THRESHOLDS:
            r = evaluate_dataset(preds, gts, iou_thr=ov, score_thr=st)
            for metric, k in (("IoU", "iou"), ("S-IoU", "s_iou")):
                v = r[k]
                out.append(dict(model=key, label=label, split=split, geometry=geom,
                                overlap=int(ov * 100), score_thr=st, metric=metric,
                                TP=v["TP"], FP=v["FP"], FN=v["FN"],
                                precision=100 * v["precision"], recall=100 * v["recall"],
                                f1=100 * v["f1"]))
    return out


if __name__ == "__main__":
    jobs = [(k, lab, sp, rp, g) for k, (lab, splits, g) in MODELOS.items()
            for sp, rp in splits.items()]
    print(len(jobs), "combinaciones modelo x split", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, r in enumerate(ex.map(job, jobs), 1):
            rows.extend(r)
            print(f"  {i}/{len(jobs)} ({len(r)} filas)", flush=True)
    dest = Path("/home/santana/Documents/docs_TFM/analisis/legacy_modelos.csv")
    pd.DataFrame(rows).to_csv(dest, index=False)
    print("->", dest, len(rows), "filas")
