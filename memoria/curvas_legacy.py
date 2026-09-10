#!/usr/bin/env python3
"""Extrae las curvas de entrenamiento y validación de los experimentos de MMDetection.

Recorre los ficheros .log.json de cada experimento, separa los registros de
entrenamiento y de validación y recoge las pérdidas y las métricas mAP de caja y de
máscara. Escribe curvas_legacy.csv, que alimenta la tabla de exploración de
arquitecturas.
"""
import json, glob
from pathlib import Path
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/Co-DETR")
OUT = Path("/home/santana/Documents/docs_TFM/analisis")

rows = []
for f in sorted(SRV.glob("*/*.log.json")):
    exp = f.parent.name
    recs = []
    for line in f.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    tr = [r for r in recs if r.get("mode") == "train"]
    va = [r for r in recs if r.get("mode") == "val"]
    if not va and not tr:
        continue
    for r in tr:
        rows.append(dict(experimento=exp, fichero=f.name, mode="train",
                         epoch=r.get("epoch"), iter=r.get("iter"), lr=r.get("lr"),
                         loss=r.get("loss"), loss_cls=r.get("loss_cls"),
                         loss_bbox=r.get("loss_bbox"), loss_iou=r.get("loss_iou"),
                         loss_mask=r.get("loss_mask"), loss_dice=r.get("loss_dice")))
    for r in va:
        rows.append(dict(experimento=exp, fichero=f.name, mode="val",
                         epoch=r.get("epoch"), iter=r.get("iter"),
                         bbox_mAP=r.get("bbox_mAP"), bbox_mAP_50=r.get("bbox_mAP_50"),
                         bbox_mAP_75=r.get("bbox_mAP_75"), bbox_mAP_s=r.get("bbox_mAP_s"),
                         bbox_mAP_m=r.get("bbox_mAP_m"), bbox_mAP_l=r.get("bbox_mAP_l"),
                         segm_mAP=r.get("segm_mAP"), segm_mAP_50=r.get("segm_mAP_50"),
                         segm_mAP_75=r.get("segm_mAP_75")))

df = pd.DataFrame(rows)
df.to_csv(OUT / "curvas_legacy.csv", index=False)
print(len(df), "filas")
print(df[df["mode"] == "val"].groupby(["experimento", "fichero"])
        .agg(n=("epoch", "size"), ep_max=("epoch", "max"),
             mAP50_max=("bbox_mAP_50", "max"), mAP_max=("bbox_mAP", "max"),
             segm50_max=("segm_mAP_50", "max")).to_string())
