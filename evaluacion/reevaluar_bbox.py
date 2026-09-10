#!/usr/bin/env python3
"""Reevalúa las ejecuciones de detección convirtiendo sus cajas en polígonos de cuatro vértices.

Los experimentos de detección exportan cajas en lugar de polígonos, así que se transforman
en rectángulos de cuatro vértices antes de aplicar la misma implementación de IoU y S-IoU
al umbral común 0,25. Las filas obtenidas sustituyen a las de esas dos ejecuciones dentro
de reeval_conf025.csv.
"""
import json, sys, os, functools
from pathlib import Path
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
sys.path.insert(0, str(SRV / "scripts"))
import siou_metrics_polygon as smp
from siou_metrics_polygon import evaluate_dataset

_orig = smp._to_shapely
@functools.lru_cache(maxsize=40_000)
def _c(t): return _orig(list(t))
smp._to_shapely = lambda c: _c(tuple(c))

EXPS = ["juniperus_btm_heterogeneous_20260824_091248",
        "juniperus_btm_homogeneous_20260824_103026"]


def caja_a_poly(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def job(path):
    j = json.load(open(path))
    preds, gts = [], []
    for im in j["images"]:
        preds.append(([caja_a_poly(b) for b in im.get("boxes", [])], im.get("scores", [])))
        gts.append([caja_a_poly(g) for g in im.get("gt", [])])
    out = []
    for ov in (0.5, 0.75):
        r = evaluate_dataset(preds, gts, iou_thr=ov, score_thr=0.25)
        for met, k in (("IoU", "iou"), ("S-IoU", "s_iou")):
            v = r[k]
            out.append(dict(experiment=Path(path).parents[2].name,
                            split=Path(path).parent.name,
                            mechanism=Path(path).stem,
                            overlap=int(ov * 100), score_thr=0.25, metric=met,
                            TP=v["TP"], FP=v["FP"], FN=v["FN"],
                            precision=100*v["precision"], recall=100*v["recall"],
                            f1=100*v["f1"]))
    return out


if __name__ == "__main__":
    jobs = [str(f) for e in EXPS
            for f in sorted((SRV / "experiments" / e).glob("predictions/*/*.json"))]
    print(len(jobs), "ficheros", flush=True)
    rows = []
    for i, f in enumerate(jobs, 1):
        rows.extend(job(f))
        print(f'  {i}/{len(jobs)}', flush=True)
    nuevo = pd.DataFrame(rows)
    dest = Path("/home/santana/Documents/docs_TFM/analisis/reeval_conf025.csv")
    viejo = pd.read_csv(dest)
    viejo = viejo[~viejo.experiment.isin(EXPS)]
    pd.concat([viejo, nuevo], ignore_index=True).to_csv(dest, index=False)
    print("corregidas", len(nuevo), "filas de deteccion")
