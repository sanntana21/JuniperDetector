#!/usr/bin/env python3
"""Mide cada anotación del conjunto de entrenamiento para elegir recortes representativos.

Recorre las etiquetas de segmentación y calcula por instancia el área, la elongación,
la compacidad, el color y el verdor del objeto y de su fondo, y el número de vecinos
próximos. Escribe _polimorfismo_stats.json, que permite localizar los extremos de
cada eje de variabilidad usados en fig_polimorfismo.py.
"""
import numpy as np, json
from pathlib import Path
from PIL import Image

BASE = Path("/home/santana/Documents/TFM")
LAB = BASE / "dataset_yolo_seg/labels/train"
IMG = BASE / "Photo_Interpretation_Data_jpg/Train/Images"

filas = []
for lf in sorted(LAB.glob("*.txt")):
    nom = lf.stem
    im = Image.open(IMG / f"{nom}.jpg").convert("RGB")
    W, H = im.size
    a = np.asarray(im).astype(np.float32)
    polis = []
    for l in lf.read_text().splitlines():
        p = l.split()
        if len(p) < 7: continue
        c = np.array([float(x) for x in p[1:]], dtype=np.float64).reshape(-1, 2) * [W, H]
        polis.append(c)
    cent = np.array([p.mean(0) for p in polis]) if polis else np.zeros((0, 2))
    for i, c in enumerate(polis):
        x = c[:, 0]; y = c[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < 20: continue
        d = c - c.mean(0)
        ev = np.linalg.eigvalsh(np.cov(d.T) + 1e-9 * np.eye(2))
        elong = float(np.sqrt(max(ev) / max(min(ev), 1e-9)))
        per = float(np.sum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)))
        compac = float(4 * np.pi * area / max(per ** 2, 1e-9))
        x0, x1 = int(max(x.min(), 0)), int(min(x.max(), W))
        y0, y1 = int(max(y.min(), 0)), int(min(y.max(), H))
        if x1 - x0 < 3 or y1 - y0 < 3: continue
        parche = a[y0:y1, x0:x1]
        R, G, B = parche[..., 0].mean(), parche[..., 1].mean(), parche[..., 2].mean()
        verdor = float(G - 0.5 * (R + B))
        # textura interna: dispersion del verdor dentro de la caja
        vmap = parche[..., 1] - 0.5 * (parche[..., 0] + parche[..., 2])
        verdor_std = float(vmap.std())
        # fondo: anillo alrededor de la caja del objeto
        m = max(8, int(0.5 * max(x1 - x0, y1 - y0)))
        bx0, bx1 = max(x0 - m, 0), min(x1 + m, W)
        by0, by1 = max(y0 - m, 0), min(y1 + m, H)
        anillo = a[by0:by1, bx0:bx1].copy()
        ix0, iy0 = x0 - bx0, y0 - by0
        anillo[iy0:iy0 + (y1 - y0), ix0:ix0 + (x1 - x0)] = np.nan
        fR = float(np.nanmean(anillo[..., 0])); fG = float(np.nanmean(anillo[..., 1]))
        fB = float(np.nanmean(anillo[..., 2]))
        fondo_verdor = float(fG - 0.5 * (fR + fB))
        fondo_std = float(np.nanstd(anillo[..., 1]))
        dist = np.linalg.norm(cent - c.mean(0), axis=1)
        vec = int(((dist > 0) & (dist < 60)).sum())
        filas.append(dict(img=nom, idx=i, area=float(area), elong=elong, compac=compac,
                          R=float(R), G=float(G), B=float(B), verdor=verdor, vecinos=vec,
                          verdor_std=verdor_std, fondo_verdor=fondo_verdor,
                          fondo_std=fondo_std, fondo_brillo=float((fR + fG + fB) / 3),
                          cx=float(c[:, 0].mean()), cy=float(c[:, 1].mean()),
                          w=float(x1 - x0), h=float(y1 - y0), n_img=len(polis)))
json.dump(filas, open("_polimorfismo_stats.json", "w"))
print("anotaciones medidas:", len(filas))
