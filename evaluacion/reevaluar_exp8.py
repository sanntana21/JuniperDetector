#!/usr/bin/env python3
"""Reevalúa las predicciones del Experimento 8 con la misma implementación de IoU y S-IoU.

Lee los volcados de predicciones del Experimento 8 y escribe reeval_exp8_conf025.csv en el
modo de umbral común 0,25, o reeval_exp8_barrido.csv en el modo de barrido, limitado a los
ficheros exportados por debajo de ese umbral. Sus cifras quedan homogéneas con el resto de
la memoria y listas para unirse a la reevaluación general.
"""
import json, sys, os, functools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/santana/Documents/TFM")
from siou_metrics_polygon import evaluate_dataset          # noqa: E402
import siou_metrics_polygon as _smp                        # noqa: E402

# memoizacion de la construccion de poligonos, ~6x sin alterar resultados
_orig = _smp._to_shapely


@functools.lru_cache(maxsize=300_000)
def _cached(coords_tuple):
    return _orig(list(coords_tuple))


_smp._to_shapely = lambda c: _cached(tuple(c))

BASE = Path("/home/santana/Documents/docs_TFM/analisis/exp8/experiments")
DEST = Path("/home/santana/Documents/docs_TFM/analisis")


def eval_file(args):
    path, thresholds, overlaps = args
    j = json.load(open(path))
    per_pred = [(im.get("polys", []), im.get("scores", [])) for im in j["images"]]
    per_gt = [im.get("gt", []) for im in j["images"]]
    p = Path(path)
    out = []
    for ov in overlaps:
        for st in thresholds:
            r = evaluate_dataset(per_pred, per_gt, iou_thr=ov, score_thr=st)
            for metric, key in (("IoU", "iou"), ("S-IoU", "s_iou")):
                v = r[key]
                out.append(dict(experiment=p.parents[2].name, split=p.parent.name,
                                mechanism=p.stem, overlap=int(ov * 100), score_thr=st,
                                metric=metric, TP=v["TP"], FP=v["FP"], FN=v["FN"],
                                precision=100 * v["precision"], recall=100 * v["recall"],
                                f1=100 * v["f1"]))
    return out


def minimo(path):
    j = json.load(open(path))
    sc = [s for im in j["images"] for s in im.get("scores", [])]
    return min(sc) if sc else 1.0


def main():
    import pandas as pd
    modo = sys.argv[1] if len(sys.argv) > 1 else "punto"
    ficheros = []
    for d in sorted(BASE.glob("juniperus_seg_*")):
        for f in sorted(d.glob("predictions/*/*.json")):
            # el backup a 0,25 duplica lo que ya esta en field_work
            if "conf025_backup" in str(f):
                continue
            ficheros.append(str(f))

    if modo == "punto":
        jobs = [(f, [0.25], (0.5, 0.75)) for f in ficheros]
        salida = "reeval_exp8_conf025.csv"
    else:
        thr = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30,
               0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
        jobs = [(f, thr, (0.5, 0.75)) for f in ficheros if minimo(f) < 0.24]
        salida = "reeval_exp8_barrido.csv"

    print(f"{len(jobs)} ficheros a evaluar en modo {modo}", flush=True)
    filas = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, res in enumerate(ex.map(eval_file, jobs), 1):
            filas.extend(res)
            print(f"  {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(DEST / salida, index=False)
    print("->", DEST / salida, len(df), "filas")


if __name__ == "__main__":
    main()
