#!/usr/bin/env python3
"""Reevalúa las predicciones crudas de cada experimento al umbral de confianza indicado.

Lee los ficheros JSON de predicciones de las ejecuciones del servidor y recalcula IoU y
S-IoU con la implementación única del proyecto. Escribe reeval_conf025.csv en el modo de
umbral común 0,25 y reeval_barrido.csv en el modo de barrido, y su función de evaluación
la importan los guiones de barrido.
"""
import json, sys, os, glob, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
sys.path.insert(0, str(SRV / "scripts"))
from siou_metrics_polygon import evaluate_dataset          # noqa: E402

# --- aceleracion: memoizacion de la construccion de poligonos shapely ---
# El matching reconstruye el mismo poligono una y otra vez (una por pareja
# prediccion-etiqueta evaluada). Cachearlo da ~6x sin alterar los resultados.
import functools                                            # noqa: E402
import siou_metrics_polygon as _smp                         # noqa: E402

_orig_to_shapely = _smp._to_shapely


@functools.lru_cache(maxsize=200_000)
def _cached_to_shapely(coords_tuple):
    return _orig_to_shapely(list(coords_tuple))


_smp._to_shapely = lambda coords: _cached_to_shapely(tuple(coords))
# ------------------------------------------------------------------------

EXPS = SRV / "experiments"


def eval_file(args):
    path, thresholds, overlaps = args
    j = json.load(open(path))
    per_pred = [(im.get("polys", []), im.get("scores", [])) for im in j["images"]]
    per_gt   = [im.get("gt", []) for im in j["images"]]
    out = []
    for ov in overlaps:
        for st in thresholds:
            r = evaluate_dataset(per_pred, per_gt, iou_thr=ov, score_thr=st)
            for metric, key in (("IoU", "iou"), ("S-IoU", "s_iou")):
                v = r[key]
                out.append(dict(
                    experiment=Path(path).parents[2].name,
                    split=Path(path).parent.name,
                    mechanism=Path(path).stem,
                    overlap=int(ov * 100), score_thr=st, metric=metric,
                    TP=v["TP"], FP=v["FP"], FN=v["FN"],
                    precision=100 * v["precision"], recall=100 * v["recall"], f1=100 * v["f1"],
                ))
    return out


def main():
    import pandas as pd
    mode = sys.argv[1] if len(sys.argv) > 1 else "punto"
    if mode == "punto":                      # un solo umbral: la comparacion oficial
        thresholds, overlaps, pattern, out = [0.25], (0.5, 0.75), "*", "reeval_conf025.csv"
    elif mode == "barrido":                  # barrido de confianza completo
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                      0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
        overlaps, pattern, out = (0.5, 0.75), "juniperus_seg_by_size_20260830_163740", "reeval_barrido.csv"
    else:
        raise SystemExit("modo: punto | barrido")

    jobs = []
    for d in sorted(EXPS.glob(pattern)):
        if not d.is_dir() or d.name.startswith("seg_smoke") or d.name.startswith("smoke"):
            continue
        for f in sorted(d.glob("predictions/*/*.json")):
            jobs.append((str(f), thresholds, overlaps))
    print(f"{len(jobs)} ficheros de predicciones a evaluar", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, res in enumerate(ex.map(eval_file, jobs), 1):
            rows.extend(res)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    dest = Path("/home/santana/Documents/docs_TFM/analisis") / out
    df.to_csv(dest, index=False)
    print("->", dest, len(df), "filas")


if __name__ == "__main__":
    main()
