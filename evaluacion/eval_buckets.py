#!/usr/bin/env python3
"""Desglosa el rendimiento por tramo de tamaño real de los arbustos anotados.

Filtra las predicciones de cada mecanismo (modelo único, expertos, ensembles y model soup)
en su umbral calibrado y las evalúa por separado en los tramos pequeño, mediano y grande del
test fotointerpretado y del test de trabajo de campo. Escribe analisis/por_bucket.csv y
resume por consola el F1 medido con S-IoU.
"""
import sys, json
from pathlib import Path
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
sys.path.insert(0, str(SRV / "scripts"))
from evaluate_by_bucket import evaluate_variant_by_bucket   # noqa: E402

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


import sys as _sys
_sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import calibrado as _cal
EXP = SRV / "experiments" / "juniperus_seg_by_size_20260830_202803"
SUMMARY = json.loads((SRV / "dataset_yolo_seg_size_buckets_summary.json").read_text())
LOW, HIGH = SUMMARY["low_threshold_px2"], SUMMARY["high_threshold_px2"]

VARIANTES = ["dense_baseline", "expert_E_small_solo", "expert_E_medium_solo",
             "expert_E_large_solo", "btm_forest_ensemble_uniform",
             "btm_forest_soup", "random_ensemble"]

rows = []
for split in ("test", "field_work"):
    for v in VARIANTES:
        f = EXP / "predictions" / split / f"{v}.json"
        if not f.exists():
            print("[aviso] falta", f.name, split); continue
        r_ = _cal.optimo(EXP.name, split, v)
        thr = r_[1] if r_ else 0.25
        data = json.loads(f.read_text())
        for im in data["images"]:
            keep = [i for i, s in enumerate(im.get("scores", [])) if s >= thr]
            im["polys"] = [im["polys"][i] for i in keep]
            im["scores"] = [im["scores"][i] for i in keep]
        iou_res, siou_res = evaluate_variant_by_bucket(data, LOW, HIGH, 0.5)
        for b in ("small", "medium", "large"):
            for met, res in (("IoU", iou_res), ("S-IoU", siou_res)):
                r = res[b]
                rows.append(dict(split=split, variante=v, bucket=b, metrica=met,
                                 TP=r["TP"], FP=r["FP"], FN=r["FN"],
                                 precision=100*r["precision"], recall=100*r["recall"],
                                 f1=100*r["f1"]))
        print(f"  {split:11s} {v}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/santana/Documents/docs_TFM/analisis/por_bucket.csv", index=False)
print(df[(df.split=="test") & (df.metrica=="S-IoU")]
      .pivot(index="variante", columns="bucket", values="f1").round(1).to_string())
