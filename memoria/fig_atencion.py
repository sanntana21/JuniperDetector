#!/usr/bin/env python3
"""Dibuja el esquema de atención completa frente a atención deformable sobre una imagen real.

Toma la escena Img_472 de la partición de entrenamiento fotointerpretada y sus polígonos anotados,
y contrasta las 169 posiciones que consulta una query con atención completa frente a los ocho
puntos que muestrea la atención deformable. Escribe la figura atencion.pdf, que acompaña a la
descripción de la familia DETR.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, FancyArrowPatch
from PIL import Image
from pathlib import Path
from estilo import TEXTWIDTH, guardar

BASE = Path("/home/santana/Documents/TFM")
NOMBRE = "Img_472"
img = np.asarray(Image.open(BASE / f"Photo_Interpretation_Data_jpg/Train/Images/{NOMBRE}.jpg"))
H, W = img.shape[:2]
polys = []
for l in (BASE / f"dataset_yolo_seg/labels/train/{NOMBRE}.txt").read_text().splitlines():
    p = l.split()
    if len(p) >= 7:
        polys.append(np.array([float(x) for x in p[1:]]).reshape(-1, 2) * [W, H])

REF = np.array([300.0, 175.0])          # posicion de referencia de la query
rng = np.random.default_rng(3)
ang = rng.permutation(np.linspace(0, 2 * np.pi, 8, endpoint=False))
rad = rng.uniform(28, 95, 8)
MUESTRAS = REF + np.c_[np.cos(ang), np.sin(ang)] * rad[:, None]

rejilla = np.array([[x, y] for y in np.linspace(24, H - 24, 13)
                            for x in np.linspace(24, W - 24, 13)])

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))
for ax, modo in zip(axes, ["completa", "deformable"]):
    ax.imshow(img)
    for c in polys:
        ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor="#00E676", linewidth=0.7))
    if modo == "completa":
        for p in rejilla:
            ax.plot([REF[0], p[0]], [REF[1], p[1]], color="#FF6D00", lw=0.25, alpha=0.5, zorder=3)
        ax.scatter(rejilla[:, 0], rejilla[:, 1], s=3, color="#FF6D00", alpha=0.9, zorder=4)
        ax.set_title("$\\it{Full\\ attention}$: 169 posiciones", fontsize=7.5, pad=3)
    else:
        ax.scatter(rejilla[:, 0], rejilla[:, 1], s=2.5, color="white", alpha=0.35, zorder=3)
        for p in MUESTRAS:
            ax.add_patch(FancyArrowPatch(REF, p, arrowstyle="-|>", mutation_scale=6,
                                         color="#FF6D00", lw=0.9, zorder=4,
                                         shrinkA=3, shrinkB=1))
        ax.scatter(MUESTRAS[:, 0], MUESTRAS[:, 1], s=14, color="#FF6D00",
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.set_title("$\\it{Deformable\\ attention}$: 8 puntos aprendidos", fontsize=7.5, pad=3)
    ax.scatter(*REF, s=42, marker="s", color="#0072B2", edgecolor="white",
               linewidth=0.8, zorder=6)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.suptitle("Dónde mira una $\\it{query}$", fontsize=9, fontweight="bold")
guardar(fig, "atencion")
