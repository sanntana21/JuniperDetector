#!/usr/bin/env python3
"""Barre el umbral de confianza hacia arriba en las dos ejecuciones de detección.

Repite el barrido de barrido_arriba.py sobre las ejecuciones que exportan cajas,
convertidas antes en rectángulos de cuatro vértices, partiendo del suelo de exportación de
cada fichero de predicciones. Escribe barrido_arriba_det.csv con las filas de S-IoU al
50 % de solapamiento, que el calibrado une al resto de barridos.
"""
import sys, os, json, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/santana/Documents/TFM")
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import reevaluar                                            # noqa: E402
from siou_metrics_polygon import evaluate_dataset            # noqa: E402


def _caja_a_poly(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def eval_cajas(args):
    """Las ejecuciones de deteccion guardan 'boxes', no 'polys'. Se convierten
    a rectangulos de cuatro vertices, que es como se calcularon sus cifras a
    umbral comun."""
    import json
    from pathlib import Path as _P
    path, thresholds, overlaps = args
    j = json.load(open(path))
    per_pred = [([_caja_a_poly(b) for b in im.get("boxes", [])], im.get("scores", []))
                for im in j["images"]]
    per_gt = [[_caja_a_poly(g) for g in im.get("gt", [])] for im in j["images"]]
    out = []
    for ov in overlaps:
        for st in thresholds:
            r = evaluate_dataset(per_pred, per_gt, iou_thr=ov, score_thr=st)
            for metric, key in (("IoU", "iou"), ("S-IoU", "s_iou")):
                v = r[key]
                out.append(dict(
                    experiment=_P(path).parents[2].name, split=_P(path).parent.name,
                    mechanism=_P(path).stem, overlap=int(ov * 100), score_thr=st,
                    metric=metric, TP=v["TP"], FP=v["FP"], FN=v["FN"],
                    precision=100 * v["precision"], recall=100 * v["recall"],
                    f1=100 * v["f1"]))
    return out

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup/experiments")
EXP8 = Path("/home/santana/Documents/docs_TFM/analisis/exp8/experiments")

RUNS = ["juniperus_btm_homogeneous_20260824_103026",
        "juniperus_btm_heterogeneous_20260824_091248"]
MECS = ["dense_baseline", "btm_forest_ensemble_uniform", "btm_forest_soup",
        "random_ensemble", "btm_forest_ensemble_weighted"]

BAJOS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22]
ALTOS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def suelo(path):
    d = json.load(open(path))
    mn = 1.0
    for im in d.get("images", []):
        sc = im.get("scores") or []
        if sc:
            mn = min(mn, min(sc))
    return mn


def localiza(run, split, mec):
    """Prefiere el volcado de exp8, que trae las reexportaciones."""
    for base in (EXP8, SRV):
        f = base / run / "predictions" / split / f"{mec}.json"
        if f.exists():
            return str(f)
    return None


def main():
    jobs = []
    for run in RUNS:
        for split in ("test", "field_work"):
            for mec in MECS:
                f = localiza(run, split, mec)
                if f is None:
                    continue
                s = suelo(f)
                thr = [t for t in BAJOS if t >= s] + ALTOS
                jobs.append((f, thr, (0.5,)))
    print(f"{len(jobs)} ficheros, {sum(len(j[1]) for j in jobs)} evaluaciones", flush=True)
    filas = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, res in enumerate(ex.map(eval_cajas, jobs), 1):
            filas.extend(res)
            print(f"  {i}/{len(jobs)}", flush=True)
    import pandas as pd
    d = pd.DataFrame(filas)
    d = d[d.metric == "S-IoU"]
    out = "/home/santana/Documents/docs_TFM/analisis/barrido_arriba_det.csv"
    d.to_csv(out, index=False)
    print("->", out, len(d), "filas")


if __name__ == "__main__":
    main()
