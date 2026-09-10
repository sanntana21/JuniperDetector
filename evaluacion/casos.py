#!/usr/bin/env python3
"""Clasifica cada instancia anotada según cómo la resolvió el modelo único.

Lee las predicciones del modelo único de una ejecución por tamaño, cada conjunto en el
umbral en que mejor rinde, y mide cuántas predicciones solapan cada anotación y qué
fracción cubren, y lo simétrico para cada predicción. Escribe casos.csv y resume las
anotaciones no detectadas o fragmentadas y los falsos positivos, base para elegir ejemplos.
"""
import json, sys, functools
from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup")
EXP = SRV / "experiments" / "juniperus_seg_by_size_20260830_202803"
OUT = Path("/home/santana/Documents/docs_TFM/analisis")
# cada conjunto se lee en el umbral en el que el modelo unico mejor rinde
CONF = {"test": 0.50, "field_work": 0.25}
THR = 0.5
# umbrales de tamano (px^2) del propio proyecto
LOW, HIGH = 197.75, 1156.25


def poly(c):
    pts = list(zip(c[0::2], c[1::2]))
    if len(pts) < 3:
        return None
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def analiza(split, mech="dense_baseline"):
    j = json.loads((EXP / "predictions" / split / f"{mech}.json").read_text())
    filas = []
    for im in j["images"]:
        preds = [poly(p) for p, s in zip(im["polys"], im["scores"]) if s >= CONF[split]]
        preds = [p for p in preds if p is not None and p.area > 0]
        gts = [poly(g) for g in im["gt"]]
        gts = [g for g in gts if g is not None and g.area > 0]

        # cada anotacion: cuantas predicciones la solapan y que fraccion cubre la union
        for k, g in enumerate(gts):
            sol = [p for p in preds if g.intersects(p)]
            cob = (g.intersection(unary_union(sol)).area / g.area) if sol else 0.0
            bucket = "small" if g.area <= LOW else ("medium" if g.area <= HIGH else "large")
            filas.append(dict(split=split, imagen=im["image"], tipo="gt", idx=k,
                              area=g.area, bucket=bucket, n_solapan=len(sol),
                              cobertura=100 * cob, detectada=cob >= THR))
        # cada prediccion: cuantas anotaciones solapa
        for k, p in enumerate(preds):
            sol = [g for g in gts if p.intersects(g)]
            prec = (p.intersection(unary_union(sol)).area / unary_union(sol).area) if sol else 0.0
            filas.append(dict(split=split, imagen=im["image"], tipo="pred", idx=k,
                              area=p.area, bucket="", n_solapan=len(sol),
                              cobertura=100 * prec, detectada=prec >= THR))
    return filas


if __name__ == "__main__":
    filas = []
    for sp in ("test", "field_work"):
        filas += analiza(sp)
        print(" ", sp, "hecho", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "casos.csv", index=False)

    g = df[df.tipo == "gt"]
    print("\n--- anotaciones por resultado ---")
    for sp in ("test", "field_work"):
        s = g[g.split == sp]
        print(f"{sp}: {len(s)} anotaciones")
        print(f"   no detectadas          : {(~s.detectada).sum():5d} ({100*(~s.detectada).mean():.1f}%)")
        print(f"   detectadas por 1 pred  : {((s.detectada)&(s.n_solapan==1)).sum():5d}")
        print(f"   fragmentadas (>=2 pred): {((s.detectada)&(s.n_solapan>=2)).sum():5d}")
        print(f"   sin ninguna prediccion : {(s.n_solapan==0).sum():5d}")
        print("   no detectadas por tamano:",
              s[~s.detectada].bucket.value_counts().to_dict())
    p = df[df.tipo == "pred"]
    print("\n--- predicciones ---")
    for sp in ("test", "field_work"):
        s = p[p.split == sp]
        print(f"{sp}: {len(s)} predicciones | falsos positivos sin solape: "
              f"{(s.n_solapan==0).sum()} | fusionan >=2 anotaciones: {(s.n_solapan>=2).sum()}")
