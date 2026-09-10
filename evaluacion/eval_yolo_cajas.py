#!/usr/bin/env python3
"""Evalúa el modelo de segmentación con geometría de caja para compararlo con los detectores.

Convierte a su caja envolvente los polígonos predichos por el modelo único y por el ensemble
Branch-Train-Merge en el test fotointerpretado y en el test de trabajo de campo, y barre el
umbral de confianza con IoU y S-IoU. Escribe analisis/yolo_cajas.csv.
"""
import sys, json, functools
from pathlib import Path
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
sys.path.insert(0, str(SRV / "scripts"))
import siou_metrics_polygon as smp
_orig = smp._to_shapely
@functools.lru_cache(maxsize=40_000)
def _c(t): return _orig(list(t))
smp._to_shapely = lambda c: _c(tuple(c))

EXP = SRV / "experiments" / "juniperus_seg_by_size_20260830_202803"
THR = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]


def caja(p):
    xs, ys = p[0::2], p[1::2]
    return [min(xs), min(ys), max(xs), min(ys), max(xs), max(ys), min(xs), max(ys)]


rows = []
for mech in ("dense_baseline", "btm_forest_ensemble_uniform"):
    for split in ("test", "field_work"):
        f = EXP / "predictions" / split / f"{mech}.json"
        j = json.loads(f.read_text())
        P = [([caja(p) for p in im["polys"]], im["scores"]) for im in j["images"]]
        G = [[caja(g) for g in im["gt"]] for im in j["images"]]
        for st in THR:
            r = smp.evaluate_dataset(P, G, 0.5, st)
            for met, k in (("IoU", "iou"), ("S-IoU", "s_iou")):
                v = r[k]
                rows.append(dict(model=f"yolo_{mech}", label="YOLOv8n-seg (denso)" if mech == "dense_baseline"
                                 else "YOLOv8n-seg (ensemble)", split=split, geometry="bbox",
                                 overlap=50, score_thr=st, metric=met,
                                 TP=v["TP"], FP=v["FP"], FN=v["FN"],
                                 precision=100*v["precision"], recall=100*v["recall"],
                                 f1=100*v["f1"]))
        print(" ", mech, split, flush=True)
pd.DataFrame(rows).to_csv("/home/santana/Documents/docs_TFM/analisis/yolo_cajas.csv", index=False)
print("listo")
