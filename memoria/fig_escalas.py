#!/usr/bin/env python3
"""Muestra la misma escena a tres resoluciones con sus anotaciones superpuestas.

Submuestrea la imagen Img_472 de la partición de entrenamiento fotointerpretada a 448, 112 y 56
píxeles y dibuja sobre cada versión los polígonos anotados. Escribe escalas.pdf, que motiva la
necesidad de trabajar a varias resoluciones para delinear los arbustos más pequeños.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from pathlib import Path
from estilo import TEXTWIDTH, guardar

BASE = Path("/home/santana/Documents/TFM")
NOMBRE = "Img_472"
img = Image.open(BASE / f"Photo_Interpretation_Data_jpg/Train/Images/{NOMBRE}.jpg")
W, H = img.size
polys = []
for l in (BASE / f"dataset_yolo_seg/labels/train/{NOMBRE}.txt").read_text().splitlines():
    p = l.split()
    if len(p) >= 7:
        polys.append(np.array([float(x) for x in p[1:]]).reshape(-1, 2) * [W, H])

ESCALAS = [(448, "448 px"), (112, "112 px"), (56, "56 px")]
fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25))
for ax, (s, etiq) in zip(axes, ESCALAS):
    # submuestreo real y vuelta al tamano de pantalla, sin interpolar
    chica = img.resize((s, s), Image.BILINEAR).resize((W, H), Image.NEAREST)
    ax.imshow(np.asarray(chica))
    for c in polys:
        ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor="#00E676", linewidth=0.8))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(etiq, fontsize=8, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.suptitle("La misma escena a tres resoluciones", fontsize=9, fontweight="bold")
guardar(fig, "escalas")
