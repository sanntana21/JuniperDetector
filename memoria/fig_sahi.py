#!/usr/bin/env python3
"""Muestra cómo cambia la escala aparente de un arbusto pequeño al trocear la imagen.

Reconstruye sobre una imagen del conjunto de entrenamiento la rejilla de recortes
solapados de evaluate_sahi_tiled.py, destaca el recorte que más arbustos pequeños
contiene y lo amplía junto a la imagen completa. Escribe la figura sahi_recortes.pdf,
que acompaña al análisis de la inferencia sobre recortes.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from estilo import TEXTWIDTH, C, guardar

RAIZ = Path("/home/santana/Documents/TFM/dataset_yolo_seg")
NOMBRE, S = "Img_607", 448
TILE, SOLAPE = 224, 0.25
UMBRAL_PEQ = 197.8

img = cv2.cvtColor(cv2.imread(str(RAIZ / "images/train" / f"{NOMBRE}.tif")),
                   cv2.COLOR_BGR2RGB)


def rejilla(w, h, t, o):
    """Identica a compute_tiles() del script del servidor."""
    p = max(1, int(t * (1 - o)))
    xs = list(range(0, max(w - t, 0) + 1, p))
    ys = list(range(0, max(h - t, 0) + 1, p))
    if not xs or xs[-1] + t < w:
        xs.append(max(w - t, 0))
    if not ys or ys[-1] + t < h:
        ys.append(max(h - t, 0))
    return xs, ys


XS, YS = rejilla(S, S, TILE, SOLAPE)

# arbustos pequenos, para elegir uno como referencia visual
peq = []
for ln in (RAIZ / "labels/train" / f"{NOMBRE}.txt").read_text().split("\n"):
    p = ln.split()
    if len(p) < 7:
        continue
    c = np.array(p[1:], float).reshape(-1, 2) * S
    a = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1))
                  - np.dot(c[:, 1], np.roll(c[:, 0], -1)))
    if a <= UMBRAL_PEQ:
        peq.append((c.mean(0), np.sqrt(a)))

# recorte destacado: el que mas arbustos pequenos contiene
mejor, x0, y0 = -1, 0, 0
for yy in YS:
    for xx in XS:
        n = sum(1 for c, _ in peq if xx <= c[0] < xx + TILE and yy <= c[1] < yy + TILE)
        if n > mejor:
            mejor, x0, y0 = n, xx, yy
# arbusto pequeno de referencia, el mas centrado dentro del recorte
cen = np.array([x0 + TILE / 2, y0 + TILE / 2])
ref, lado = min(((c, l) for c, l in peq
                 if x0 <= c[0] < x0 + TILE and y0 <= c[1] < y0 + TILE),
                key=lambda t: np.hypot(*(t[0] - cen)))

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.65))

axes[0].imshow(img)
for yy in YS:
    for xx in XS:
        axes[0].add_patch(Rectangle((xx, yy), TILE, TILE, fill=False,
                                    edgecolor="white", lw=0.7, alpha=0.85))
axes[0].add_patch(Rectangle((x0, y0), TILE, TILE, fill=False,
                            edgecolor=C["ensemble"], lw=1.8))
axes[0].add_patch(Circle(ref, lado * 1.9, fill=False, edgecolor=C["soup"], lw=1.2))
axes[0].set_title(f"Imagen completa, {S}$\\times${S}\n"
                  f"{len(XS) * len(YS)} recortes de {TILE}$\\times${TILE}", fontsize=6.6)
axes[0].set_xlim(0, S); axes[0].set_ylim(S, 0)

axes[1].imshow(img[y0:y0 + TILE, x0:x0 + TILE])
axes[1].add_patch(Circle(ref - [x0, y0], lado * 1.9, fill=False,
                         edgecolor=C["soup"], lw=1.2))
axes[1].set_title(f"El recorte destacado, {TILE}$\\times${TILE}\n"
                  "el modelo lo reescala a 448", fontsize=6.6)
axes[1].set_xlim(0, TILE); axes[1].set_ylim(TILE, 0)

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle("El slicing duplica la escala aparente del arbusto",
             fontsize=9, fontweight="bold")
guardar(fig, "sahi_recortes")
