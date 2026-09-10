#!/usr/bin/env python3
"""Dibuja la distribución geográfica de las imágenes y su reparto en las dos regiones.

Lee el centro de cada imagen de los tags GeoTIFF del conjunto, reproduce el
agrupamiento single-linkage y el reparto por tamaño de cluster de
bucket_images_by_geography.py y añade una miniatura por cada agrupación grande.
Escribe la figura regiones.pdf, que documenta el criterio de especialización
geográfica.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path
from scipy.cluster.hierarchy import fcluster, linkage
from estilo import TEXTWIDTH, C, guardar

RAIZ = Path("/home/santana/Documents/TFM/dataset_yolo_seg/images")
CORTE = 0.02          # grados, el mismo --cluster-distance del script original
N_REG = 2
COLOR = [C["denso"], C["ensemble"]]
MINI = 110            # lado del thumbnail en pixeles

# 1. centro geografico de cada imagen, leido de los tags GeoTIFF
reg = []
for d in sorted(RAIZ.iterdir()):
    for f in sorted(d.glob("*")):
        im = Image.open(f)
        if 33922 not in im.tag_v2 or 33550 not in im.tag_v2:
            continue
        tie, sc = im.tag_v2[33922], im.tag_v2[33550]
        w, h = im.size
        reg.append((f, tie[3] + w * sc[0] / 2, tie[4] - h * sc[1] / 2))

lon = np.array([r[1] for r in reg])
lat = np.array([r[2] for r in reg])

# 2. clusters naturales y reparto LPT (mayor a menor, siempre a la region con menos)
cl = fcluster(linkage(np.column_stack([lon, lat]), method="single"),
              t=CORTE, criterion="distance")
tam = {c: int((cl == c).sum()) for c in set(cl)}
region_de, cuenta = {}, [0] * N_REG
for c in sorted(tam, key=lambda c: -tam[c]):
    r = min(range(N_REG), key=lambda i: cuenta[i])
    region_de[c] = r
    cuenta[r] += tam[c]
region = np.array([region_de[c] for c in cl])

# 3. una miniatura por cada cluster grande, la imagen mas cercana a su centro
GRANDES = sorted((c for c in tam if tam[c] >= 100), key=lambda c: lon[cl == c].mean())
BANDA = 0.055        # las miniaturas van en una fila por encima de los datos

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.75))
for r in range(N_REG):
    m = region == r
    ax.scatter(lon[m], lat[m], s=5, color=COLOR[r], alpha=0.75, linewidths=0,
               label=f"Región {r} · {cuenta[r]} imágenes", zorder=3)

ARRIBA = lat.max() + BANDA
for c in GRANDES:
    m = cl == c
    cx, cy = lon[m].mean(), lat[m].mean()
    i = np.argmin((lon - cx) ** 2 + (lat - cy) ** 2 + np.where(m, 0, 1e6))
    mini = Image.open(reg[i][0]).convert("RGB").resize((MINI, MINI), Image.LANCZOS)
    ax.plot([lon[i], lon[i]], [lat[i], ARRIBA], lw=0.6, color=C["gris"], zorder=2)
    ab = AnnotationBbox(OffsetImage(np.asarray(mini), zoom=0.26),
                        (lon[i], ARRIBA), frameon=True, pad=0.1,
                        bboxprops=dict(edgecolor=COLOR[region[i]], lw=0.9))
    ax.add_artist(ab)

ax.set_xlabel("Longitud (°)")
ax.set_ylabel("Latitud (°)")
ax.set_ylim(lat.min() - 0.018, lat.max() + 0.105)
ax.set_xlim(lon.min() - 0.03, lon.max() + 0.03)
ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
ax.legend(loc="lower right", markerscale=2.2)
ax.set_title("Distribución geográfica de las imágenes y reparto en regiones")
guardar(fig, "regiones")
