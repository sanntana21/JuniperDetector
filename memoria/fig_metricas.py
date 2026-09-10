#!/usr/bin/env python3
"""Compara IoU y S-IoU sobre los tres casos que separan a las dos métricas.

Parte de anotaciones reales del conjunto de entrenamiento, construye sobre ellas las
predicciones que representan correspondencia one-to-one, fragmentación y fusión, y
calcula en cada caso el mejor IoU individual y los dos lados de S-IoU. Escribe la
figura metricas_iou.pdf, que justifica el uso de S-IoU en la evaluación.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from estilo import TEXTWIDTH, guardar

BASE = Path("/home/santana/Documents/TFM")
LAB = BASE / "dataset_yolo_seg/labels/train"
IMG = BASE / "Photo_Interpretation_Data_jpg/Train/Images"
VERDE, NARANJA = "#00C853", "#FF6D00"


def carga(nombre):
    im = Image.open(IMG / f"{nombre}.jpg").convert("RGB")
    W, H = im.size
    polis = []
    for l in (LAB / f"{nombre}.txt").read_text().splitlines():
        p = l.split()
        if len(p) < 7:
            continue
        polis.append(Polygon(np.array([float(x) for x in p[1:]]).reshape(-1, 2) * [W, H]).buffer(0))
    return np.asarray(im), polis


def iou(p, l):
    return p.intersection(l).area / p.union(l).area


def siou_pred(p, etiquetas):
    """Ecuacion 4.2: area del solape con la union de etiquetas coincidentes,
    normalizada por el area de esa union."""
    coin = [l for l in etiquetas if p.intersects(l) and p.intersection(l).area > 0]
    if not coin:
        return 0.0
    u = unary_union(coin)
    return p.intersection(u).area / u.area


def siou_lab(l, predicciones):
    """Ecuacion 4.3: area de la etiqueta cubierta por las predicciones."""
    coin = [p for p in predicciones if p.intersects(l) and p.intersection(l).area > 0]
    if not coin:
        return 0.0
    return l.intersection(unary_union(coin)).area / l.area


# ---------------------------------------------------------------- los tres casos
arr192, pol192 = carga("Img_192")
arr306, pol306 = carga("Img_306")

# (a) correspondencia one-to-one: la prediccion es la etiqueta desplazada y encogida
eti_a = [pol192[4]]
pre_a = [affinity.scale(affinity.translate(eti_a[0], 4.5, 3.5), 0.90, 0.90)]

# (b) fragmentacion: tres trozos de la misma etiqueta
e = pol192[4]
x0, y0, x1, y1 = e.bounds
cortes = []
for k in range(3):
    banda = Polygon([(x0 - 5, y0 + k * (y1 - y0) / 3), (x1 + 5, y0 + k * (y1 - y0) / 3),
                     (x1 + 5, y0 + (k + 1) * (y1 - y0) / 3), (x0 - 5, y0 + (k + 1) * (y1 - y0) / 3)])
    t = e.intersection(banda).buffer(-1.2)
    if not t.is_empty:
        cortes.append(t if t.geom_type == "Polygon" else max(t.geoms, key=lambda g: g.area))
eti_b, pre_b = [e], cortes

# (c) fusion: una sola prediccion envuelve a tres etiquetas contiguas
cx, cy = 205.6, 302.6
vecinas = sorted([g for g in pol306 if 250 < g.area < 3000],
                 key=lambda p: (p.centroid.x - cx) ** 2 + (p.centroid.y - cy) ** 2)[:3]
eti_c = vecinas
pre_c = [unary_union([v.buffer(3.5) for v in vecinas]).convex_hull.buffer(-1.0)]

CASOS = [
    ("(a) Correspondencia one-to-one", arr192, eti_a, pre_a),
    ("(b) Fragmentación", arr192, eti_b, pre_b),
    ("(c) Fusión", arr306, eti_c, pre_c),
]


def dibuja(ax, arr, etiquetas, predicciones, titulo):
    todo = unary_union([g.buffer(0) for g in etiquetas + predicciones])
    x0, y0, x1, y1 = todo.bounds
    lado = max(x1 - x0, y1 - y0) * 1.22
    cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
    H, W = arr.shape[:2]
    ix0 = int(np.clip(cx_ - lado / 2, 0, W - lado)); iy0 = int(np.clip(cy_ - lado / 2, 0, H - lado))
    L = int(lado)
    ax.imshow(arr[iy0:iy0 + L, ix0:ix0 + L])
    for g, col, lw, ls in [(etiquetas, VERDE, 1.5, "-"), (predicciones, NARANJA, 1.5, "--")]:
        for p in g:
            c = np.array(p.exterior.coords) - [ix0, iy0]
            ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor=col, linewidth=lw, linestyle=ls))
    ax.set_xlim(0, L); ax.set_ylim(L, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("#BBBBBB"); s.set_linewidth(0.5)
    ax.set_title(titulo, fontsize=7, pad=3)


plt.rcParams["figure.constrained_layout.use"] = False
fig = plt.figure(figsize=(TEXTWIDTH, 2.95))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.40], hspace=0.06, wspace=0.07,
                      left=0.015, right=0.985, top=0.815, bottom=0.02)


def coma(v):
    return f"{v:.2f}".replace(".", ",")


for j, (titulo, arr, eti, pre) in enumerate(CASOS):
    dibuja(fig.add_subplot(gs[0, j]), arr, eti, pre, titulo)
    mejor_iou = max(iou(p, l) for p in pre for l in eti)
    sp = min(siou_pred(p, eti) for p in pre)
    sl = min(siou_lab(l, pre) for l in eti)
    tx = fig.add_subplot(gs[1, j])
    tx.axis("off")
    filas = [("Mejor IoU individual", mejor_iou, mejor_iou >= 0.5),
             ("S-IoU, lado de la etiqueta", sl, sl >= 0.5),
             ("S-IoU, lado de la predicción", sp, sp >= 0.5)]
    for k, (etq, val, ok) in enumerate(filas):
        y = 0.90 - k * 0.30
        col = "#1B7F3B" if ok else "#B3261E"
        tx.text(0.02, y, etq, ha="left", va="top", fontsize=5.2,
                transform=tx.transAxes, color="#333333")
        tx.text(0.98, y, coma(val), ha="right", va="top", fontsize=5.3,
                transform=tx.transAxes, color=col, fontweight="bold")

fig.text(0.5, 0.895, "Anotación en verde continuo, predicción en naranja discontinuo.  "
                     "En rojo, los valores por debajo del umbral de 0,50",
         ha="center", fontsize=5.2, color="#555555")
fig.text(0.5, 0.862, "El lado de la etiqueta decide si el arbusto se da por detectado.  "
                     "El lado de la predicción decide si esa predicción es un falso positivo",
         ha="center", fontsize=5.2, color="#555555")
fig.suptitle("Qué mide IoU y qué mide S-IoU", fontsize=9, fontweight="bold", y=0.985)
guardar(fig, "metricas_iou")
