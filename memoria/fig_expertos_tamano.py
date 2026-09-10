#!/usr/bin/env python3
"""Dibuja la misma escena anotada tal como la ve cada uno de los tres expertos de tamaño.

Carga una imagen del conjunto de entrenamiento y sus polígonos, los reparte en
pequeño, mediano y grande según los cuantiles 25 y 75 del área anotada y muestra en
cada panel solo los del rango correspondiente. Escribe la figura expertos_tamano.pdf,
que ilustra el criterio de especialización por tamaño.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from estilo import TEXTWIDTH, C, guardar

RAIZ = Path("/home/santana/Documents/TFM/dataset_yolo_seg")
NOMBRE = "Img_607"
S = 448
UMBRAL = (197.8, 1156.2)        # cuantiles 25/75 del area anotada
COLOR = [C["ensemble"], C["experto2"], C["soup"]]
TITULO = ["Experto pequeño\n$\\leq$ 197,8 px$^2$",
          "Experto mediano\n197,8 – 1.156,2 px$^2$",
          "Experto grande\n$>$ 1.156,2 px$^2$"]

img = cv2.cvtColor(cv2.imread(str(RAIZ / "images/train" / f"{NOMBRE}.tif")),
                   cv2.COLOR_BGR2RGB)

grupos = [[], [], []]
for ln in (RAIZ / "labels/train" / f"{NOMBRE}.txt").read_text().split("\n"):
    p = ln.split()
    if len(p) < 7:
        continue
    c = np.array(p[1:], float).reshape(-1, 2) * S
    a = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1))
                  - np.dot(c[:, 1], np.roll(c[:, 0], -1)))
    grupos[0 if a <= UMBRAL[0] else (1 if a <= UMBRAL[1] else 2)].append(c)

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15))
for i, ax in enumerate(axes):
    ax.imshow(img)
    for c in grupos[i]:
        ax.add_patch(Polygon(c, closed=True, facecolor=COLOR[i], alpha=0.55,
                             edgecolor=COLOR[i], linewidth=1.1))
    ax.set_title(f"{TITULO[i]}\n{len(grupos[i])} anotaciones", fontsize=6.3, pad=3)
    ax.set_xlim(0, S); ax.set_ylim(S, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle("La misma escena vista por cada experto de tamaño",
             fontsize=9, fontweight="bold")
guardar(fig, "expertos_tamano")
