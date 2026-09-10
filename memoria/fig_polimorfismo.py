#!/usr/bin/env python3
"""Reúne recortes de anotaciones reales que ilustran los seis ejes de variabilidad del enebro.

Compone una rejilla de tres recortes por eje sobre imágenes del conjunto de
entrenamiento, con los polígonos de la anotación superpuestos; los casos proceden de
las medidas calculadas por _sel_polimorfismo.py. Escribe la figura polimorfismo.pdf,
que documenta la variabilidad del objeto a delinear.
"""
import sys, json
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from pathlib import Path
from estilo import TEXTWIDTH, guardar

BASE = Path("/home/santana/Documents/TFM")
LAB = BASE / "dataset_yolo_seg/labels/train"
IMG = BASE / "Photo_Interpretation_Data_jpg/Train/Images"
VERDE = "#00E676"

# (titulo del eje, [(imagen, indice de anotacion, pie), ...])
EJES = [
    ("Morfología", [
        ("Img_79", 0, "hemisférica"),
        ("Img_303", 1, "en franja"),
        ("Img_296", 70, "lobulada"),
    ]),
    ("Tamaño", [
        ("Img_671", 1, "75 px$^2$"),
        ("Img_149", 0, "508 px$^2$"),
        ("Img_518", 0, "36.207 px$^2$"),
    ]),
    ("Densidad de follaje", [
        ("Img_301", 22, "compacto"),
        ("Img_470", 3, "intermedio"),
        ("Img_537", 5, "aclarado"),
    ]),
    ("Densidad de individuos", [
        ("Img_175", 0, "aislado"),
        ("Img_572", 11, "en grupo"),
        ("Img_706", 45, "en colonia"),
    ]),
    ("Fondo", [
        ("Img_305", 15, "suelo desnudo"),
        ("Img_430", 0, "roca y sombra"),
        ("Img_309", 2, "muy texturizado"),
    ]),
    ("Color", [
        ("Img_593", 5, "verde intenso"),
        ("Img_571", 1, "verde apagado"),
        ("Img_247", 7, "pardo"),
    ]),
]

CACHE = {}


def poligonos(nombre):
    if nombre in CACHE:
        return CACHE[nombre]
    im = Image.open(IMG / f"{nombre}.jpg").convert("RGB")
    W, H = im.size
    polis = []
    for l in (LAB / f"{nombre}.txt").read_text().splitlines():
        p = l.split()
        if len(p) < 7:
            continue
        polis.append(np.array([float(x) for x in p[1:]]).reshape(-1, 2) * [W, H])
    CACHE[nombre] = (np.asarray(im), polis)
    return CACHE[nombre]


def recorte(ax, nombre, idx, pie):
    arr, polis = poligonos(nombre)
    H, W = arr.shape[:2]
    c = polis[idx]
    x0, x1 = c[:, 0].min(), c[:, 0].max()
    y0, y1 = c[:, 1].min(), c[:, 1].max()
    lado = max(x1 - x0, y1 - y0)
    lado = float(np.clip(lado * 2.1, 56, min(W, H)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ix0 = int(np.clip(cx - lado / 2, 0, W - lado))
    iy0 = int(np.clip(cy - lado / 2, 0, H - lado))
    L = int(lado)
    ax.imshow(arr[iy0:iy0 + L, ix0:ix0 + L])
    for p in polis:
        q = p - [ix0, iy0]
        if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > L or q[:, 1].min() > L:
            continue
        ax.add_patch(MplPoly(q, closed=True, fill=False, edgecolor=VERDE, linewidth=0.75))
    ax.set_xlim(0, L); ax.set_ylim(L, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_color("#BBBBBB"); s.set_linewidth(0.5)
    ax.set_xlabel(pie, fontsize=5.4, labelpad=1.5, color="#444444")


plt.rcParams["figure.constrained_layout.use"] = False
fig = plt.figure(figsize=(TEXTWIDTH, 4.9))
outer = fig.add_gridspec(3, 2, hspace=0.17, wspace=0.07,
                         left=0.012, right=0.988, top=0.955, bottom=0.012)

for k, (titulo, casos) in enumerate(EJES):
    fila, col = divmod(k, 2)
    sub = outer[fila, col].subgridspec(2, 3, height_ratios=[0.16, 1],
                                       hspace=0.10, wspace=0.05)
    banda = fig.add_subplot(sub[0, :])
    banda.set_facecolor("#E8F0DE")
    banda.set_xticks([]); banda.set_yticks([]); banda.grid(False)
    for sp in banda.spines.values():
        sp.set_visible(False)
    banda.text(0.5, 0.42, titulo, ha="center", va="center",
               fontsize=7.2, fontweight="bold", transform=banda.transAxes)
    for j, (nom, idx, pie) in enumerate(casos):
        recorte(fig.add_subplot(sub[1, j]), nom, idx, pie)

fig.suptitle("Ejes de variabilidad del enebro en el conjunto de entrenamiento",
             fontsize=9, fontweight="bold", y=0.995)
guardar(fig, "polimorfismo")
