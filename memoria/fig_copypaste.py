#!/usr/bin/env python3
"""Reproduce el aumento por copy-paste sobre una escena real del conjunto de entrenamiento.

Recorre las imágenes y etiquetas de dataset_yolo_seg para formar un banco de recortes del rango de
arbustos pequeños y pega tres de ellos sobre la escena Img_540, con el mismo umbral de área y el
mismo suavizado de borde que emplea el script del proyecto. Escribe copypaste.pdf, con la imagen
original, la imagen aumentada y un detalle ampliado del pegado.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Circle
from pathlib import Path
from estilo import TEXTWIDTH, guardar

RAIZ = Path("/home/santana/Documents/TFM/dataset_yolo_seg")
IMGD, LBLD = RAIZ / "images/train", RAIZ / "labels/train"
UMBRAL = 197.75          # mismo umbral de rango pequeno que usa el proyecto
DESTINO = "Img_540"      # escena con pocos ejemplares, para que se vea el efecto
VERDE, NARANJA = "#00E676", "#FF6D00"


def poligonos(stem):
    img = cv2.imread(str(IMGD / f"{stem}.tif"))
    h, w = img.shape[:2]
    out = []
    for l in (LBLD / f"{stem}.txt").read_text().splitlines():
        p = l.split()
        if len(p) >= 7:
            c = np.array([float(x) for x in p[1:]]).reshape(-1, 2) * [w, h]
            out.append(c)
    return img, out


def area(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --- 1. banco de recortes del rango pequeno, como extract_small_patches()
parches = []
for lbl in sorted(LBLD.glob("*.txt"))[:120]:
    if lbl.stem == DESTINO:
        continue
    img, polys = poligonos(lbl.stem)
    for c in polys:
        if area(c) > UMBRAL:
            continue
        x0, y0 = c[:, 0].min(), c[:, 1].min()
        x1, y1 = c[:, 0].max(), c[:, 1].max()
        pad = 4
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(img.shape[1], int(x1 + pad)), min(img.shape[0], int(y1 + pad))
        if cx1 - cx0 < 6 or cy1 - cy0 < 6:
            continue
        crop = img[cy0:cy1, cx0:cx1].copy()
        rel = c - [cx0, cy0]
        m = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(m, [rel.astype(np.int32)], 255)
        parches.append(dict(crop=crop, mask=m, rel=rel))
print(f"{len(parches)} recortes del rango pequeño disponibles")

# --- 2. pegado, como paste_patch()
destino, originales = poligonos(DESTINO)
aug = destino.copy()
rng = np.random.RandomState(7)
pegados = []
for p in [parches[i] for i in rng.choice(len(parches), 3, replace=False)]:
    ch, cw = p["crop"].shape[:2]
    for _ in range(40):
        px, py = rng.randint(0, aug.shape[1] - cw), rng.randint(0, aug.shape[0] - ch)
        caja = (px, py, px + cw, py + ch)
        choca = False
        for c in originales:
            b = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
            ix = max(0, min(caja[2], b[2]) - max(caja[0], b[0]))
            iy = max(0, min(caja[3], b[3]) - max(caja[1], b[1]))
            if ix * iy > 0.1 * cw * ch:
                choca = True
                break
        if choca:
            continue
        alfa = cv2.GaussianBlur(p["mask"].astype(np.float32) / 255.0, (5, 5), 0)[:, :, None]
        reg = aug[py:py + ch, px:px + cw].astype(np.float32)
        aug[py:py + ch, px:px + cw] = (reg * (1 - alfa) + p["crop"].astype(np.float32) * alfa).astype(np.uint8)
        pegados.append(p["rel"] + [px, py])
        break

# --- 3. figura
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(TEXTWIDTH, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.06)

for k, (im, polys, extra, tit) in enumerate([
        (destino, originales, [], "(a) Imagen original"),
        (aug, originales, pegados, "(b) Tras el pegado")]):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(rgb(im))
    for c in polys:
        ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor=VERDE, linewidth=0.9))
    for c in extra:
        ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor=NARANJA, linewidth=0.9))
        # circulo localizador: los parches son pequenos y a tamano de impresion
        # pasan desapercibidos sin una marca que guie la vista
        ax.add_patch(Circle((c[:, 0].mean(), c[:, 1].mean()), 26, fill=False,
                            edgecolor=NARANJA, linewidth=0.7, linestyle=(0, (3, 2))))
    ax.set_title(tit, fontsize=7.5, pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

# detalle ampliado del primer parche pegado
c = pegados[0]
cx, cy = c[:, 0].mean(), c[:, 1].mean()
r = 38
ax = fig.add_subplot(gs[0, 2])
ax.imshow(rgb(aug))
ax.add_patch(MplPoly(c, closed=True, fill=False, edgecolor=NARANJA, linewidth=1.2))
ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy + r, cy - r)
ax.set_title("(c) Detalle del pegado", fontsize=7.5, pad=3)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)

fig.suptitle("Aumento por copy-paste", fontsize=9, fontweight="bold")
guardar(fig, "copypaste")
