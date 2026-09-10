#!/usr/bin/env python3
"""Barre el umbral de confianza completo en la única ejecución exportada a confianza muy baja.

Esa ejecución conserva predicciones por debajo de 0,25, de modo que permite recorrer todo
el rango para el modelo único, los ensembles, el model merging y los tres expertos de
tamaño en ambos conjuntos de evaluación. Escribe barrido_confianza.csv con IoU y S-IoU al
50 % de solapamiento, réplica del barrido de confianza del trabajo de referencia.
"""
import json, sys, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
sys.path.insert(0, str(SRV / "scripts"))
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


# El experimento con predicciones exportadas a confianza muy baja es el unico
# que permite barrer por debajo de 0.25 en el split de test.
EXP_TEST = SRV / "experiments" / "juniperus_seg_by_size_20260830_163740"
MECANISMOS = ["dense_baseline", "btm_forest_ensemble_uniform",
              "btm_forest_soup", "random_ensemble",
              "expert_E_small_solo", "expert_E_medium_solo", "expert_E_large_solo"]
THR = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
       0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]


def job(args):
    path, mech = args
    j = json.load(open(path))
    preds = [(im.get("polys", []), im.get("scores", [])) for im in j["images"]]
    gts = [im.get("gt", []) for im in j["images"]]
    smax = max((s for _, ss in preds for s in ss), default=0)
    smin = min((s for _, ss in preds for s in ss), default=1)
    out = []
    for st in THR:
        if st > smax:
            continue
        r = evaluate_dataset(preds, gts, iou_thr=0.5, score_thr=st)
        for met, k in (("IoU", "iou"), ("S-IoU", "s_iou")):
            v = r[k]
            out.append(dict(mechanism=mech, split=Path(path).parent.name,
                            score_thr=st, metric=met, score_min_exportado=smin,
                            TP=v["TP"], FP=v["FP"], FN=v["FN"],
                            precision=100*v["precision"], recall=100*v["recall"],
                            f1=100*v["f1"]))
    return out


if __name__ == "__main__":
    jobs = []
    for split in ("test", "field_work"):
        for m in MECANISMOS:
            f = EXP_TEST / "predictions" / split / f"{m}.json"
            if f.exists():
                jobs.append((str(f), m))
    print(len(jobs), "combinaciones", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, r in enumerate(ex.map(job, jobs), 1):
            rows.extend(r)
            print(f"  {i}/{len(jobs)}", flush=True)
    dest = Path("/home/santana/Documents/docs_TFM/analisis/barrido_confianza.csv")
    pd.DataFrame(rows).to_csv(dest, index=False)
    print("->", dest, len(rows))
