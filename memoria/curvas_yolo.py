#!/usr/bin/env python3
"""Consolida en un único CSV las curvas de entrenamiento de todas las ejecuciones YOLO.

Recorre los results.csv de los directorios de experimentos, descarta las pruebas de
humo y concatena las épocas etiquetadas por experimento y ejecución. Escribe
curvas_yolo.csv e imprime un resumen de épocas y de mAP50 de máscara por ejecución.
"""
from pathlib import Path
import pandas as pd

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup/experiments")
OUT = Path("/home/santana/Documents/docs_TFM/analisis")

frames = []
for f in sorted(SRV.glob("*/*/results.csv")):
    exp, run = f.parents[1].name, f.parent.name
    if exp.startswith(("smoke", "seg_smoke")):
        continue
    try:
        df = pd.read_csv(f)
    except Exception:
        continue
    df.columns = [c.strip() for c in df.columns]
    df["experiment"], df["run"] = exp, run
    frames.append(df)

if frames:
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "curvas_yolo.csv", index=False)
    print(len(all_df), "filas;", all_df.experiment.nunique(), "experimentos")
    print(all_df.groupby(["experiment", "run"]).agg(
        epocas=("epoch", "max"),
        mAP50_M_final=("metrics/mAP50(M)", "last"),
        mAP50_M_max=("metrics/mAP50(M)", "max")).to_string())
