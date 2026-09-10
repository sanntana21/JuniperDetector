#!/usr/bin/env python3
"""Genera las tres figuras del bloque de expertos: participación, curvas de entrenamiento y slicing.

Lee participacion.csv y curvas_yolo.csv del directorio de análisis, junto con los volcados JSON de
la inferencia con slicing, para los que emplea valores de respaldo cuando no están disponibles.
Escribe btm_participacion.pdf, con cuántos expertos detectan cada arbusto; btm_bug_curvas.pdf, con
el efecto de filtrar las anotaciones al entrenar los expertos por tamaño; y btm_sahi.pdf, con el F1
en el test de trabajo de campo medido con IoU y con S-IoU.
"""
import sys, json
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from estilo import TEXTWIDTH, C, guardar

A = "/home/santana/Documents/docs_TFM/analisis"
SRV = "/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup"

# ============================================================ 1. Participacion
part = pd.read_csv(f"{A}/participacion.csv")
CASOS = [
    ("juniperus_seg_homogeneous_20260905_105741",   "Data augmentation\nhomogéneo",   3),
    ("juniperus_seg_heterogeneous_20260905_111758", "Data augmentation\nheterogéneo", 3),
    ("juniperus_seg_by_region_20260831_073029",     "Región geográfica",                         2),
    ("juniperus_seg_by_size_20260830_202803",       "Tamaño de arbusto",                         3),
]
COLS = ["#D6D6D6", "#9ECAE1", "#4292C6", "#08519C"]
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 3.1), sharey=True)
for ax, split, tit in zip(axes, ["test", "field_work"], ["Test (PI)", "Campo (FW)"]):
    labels, base = [], np.zeros(len(CASOS))
    datos = []
    for exp, lab, nexp in CASOS:
        r = part[(part.experiment == exp) & (part.split == split)]
        h = json.loads(r.iloc[0].agreement_histogram) if len(r) else {}
        tot = sum(h.values()) or 1
        datos.append([100 * h.get(str(k), 0) / tot for k in range(4)])
        labels.append(lab)
    datos = np.array(datos)
    y = np.arange(len(CASOS))
    left = np.zeros(len(CASOS))
    for k in range(4):
        ax.barh(y, datos[:, k], left=left, color=COLS[k], height=0.62,
                label=f"{k} expertos" if k != 1 else "1 experto")
        left += datos[:, k]
    # el porcentaje del acuerdo maximo de cada fila, que es la cifra que cita el texto
    for i, (_, _, nexp) in enumerate(CASOS):
        v = datos[i, nexp]
        if v < 3:
            continue
        ax.text(left[i] - v / 2, y[i], f"{v:.1f}".replace(".", ","), va="center",
                ha="center", fontsize=5.6, color="white", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(0, 100); ax.set_xlabel("Arbustos anotados (%)")
    ax.set_title(tit, fontsize=8)
    ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.10), fontsize=7)
fig.suptitle("¿Cuántos expertos detectan cada arbusto?", fontsize=9, fontweight="bold")
guardar(fig, "btm_participacion")

# ================================= 2. Curvas de entrenamiento: bug vs corregido
cur = pd.read_csv(f"{A}/curvas_yolo.csv")
EXPERTOS = {"E_small": ("E pequeño", "#56B4E9"), "E_medium": ("E mediano", "#E69F00"),
            "E_large": ("E grande", "#D55E00")}
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.1), sharey=True)
for ax, (exp, tit) in zip(axes, [
        ("juniperus_seg_by_size_20260828_072323", "Cada experto ve todo"),
        ("juniperus_seg_by_size_20260830_163740", "Cada experto ve su rango")]):
    for run, (lab, col) in EXPERTOS.items():
        s = cur[(cur.experiment == exp) & (cur.run == run)].sort_values("epoch")
        if s.empty:
            continue
        ax.plot(s["epoch"], 100 * s["metrics/mAP50(M)"], label=lab, color=col)
    ax.set_xlabel("Época"); ax.set_title(tit, fontsize=8); ax.set_ylim(0, 100)
axes[0].set_ylabel("mAP@50 de máscara\nen validación (%)")
axes[1].legend(loc="lower right")
fig.suptitle("Efecto de filtrar las anotaciones sobre el entrenamiento",
             fontsize=9, fontweight="bold")
guardar(fig, "btm_bug_curvas")

# ============================================================ 3. SAHI
import glob
filas = []
NOM = {"sahi_dense_baseline_field_work_tile224_ov0.25.json": "conf. 0.25\nsin filtro de borde",
       "sahi_dense_baseline_field_work_tile224_ov0.25_conf0.4_edge15.json": "conf. 0.40\nfiltro amplio",
       "sahi_dense_baseline_field_work_tile224_ov0.25_conf0.4_edge5.json": "conf. 0.40\nfiltro estrecho"}
for f in sorted(glob.glob(f"{SRV}/sahi_results/*.json")):
    j = json.load(open(f))
    nom = NOM.get(f.split("/")[-1])
    if not nom:
        continue
    for modo in ("baseline", "tiled"):
        for met in ("iou", "s_iou"):
            filas.append(dict(cfg=nom, modo=modo, metrica="IoU" if met == "iou" else "S-IoU",
                              f1=100 * j[modo][met]["f1"], tp=j[modo][met]["TP"],
                              fn=j[modo][met]["FN"]))
if not filas:
    # El volcado del servidor no esta disponible: se usan los valores de respaldo,
    # los mismos que emplea tablas.py para tab:exp5_sahi.
    RESPALDO = [
        ("conf. 0.25\nsin filtro de borde", 67.45, 62.06, 79.26, 64.14),
        ("conf. 0.40\nfiltro amplio",       67.46, 69.18, 75.36, 69.41),
        ("conf. 0.40\nfiltro estrecho",     67.46, 69.94, 75.36, 70.25),
    ]
    for nom, bi, ti, bs, ts in RESPALDO:
        filas.append(dict(cfg=nom, modo="baseline", metrica="IoU",   f1=bi, tp=0, fn=0))
        filas.append(dict(cfg=nom, modo="tiled",    metrica="IoU",   f1=ti, tp=0, fn=0))
        filas.append(dict(cfg=nom, modo="baseline", metrica="S-IoU", f1=bs, tp=0, fn=0))
        filas.append(dict(cfg=nom, modo="tiled",    metrica="S-IoU", f1=ts, tp=0, fn=0))

s = pd.DataFrame(filas)
orden = ["conf. 0.25\nsin filtro de borde", "conf. 0.40\nfiltro amplio", "conf. 0.40\nfiltro estrecho"]
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2), sharey=True)
for ax, met in zip(axes, ["IoU", "S-IoU"]):
    sub = s[s.metrica == met]
    x = np.arange(len(orden)); w = 0.36
    b1 = ax.bar(x - w/2, [sub[(sub.cfg == c) & (sub.modo == "baseline")].f1.iloc[0] for c in orden],
                w, label="Sin slicing", color=C["denso"])
    b2 = ax.bar(x + w/2, [sub[(sub.cfg == c) & (sub.modo == "tiled")].f1.iloc[0] for c in orden],
                w, label="Slicing (SAHI)", color=C["ensemble"])
    for bs in (b1, b2):
        ax.bar_label(bs, fmt="%.1f", fontsize=6, padding=1)
    ax.set_xticks(x); ax.set_xticklabels(orden, fontsize=6.5)
    ax.set_title(met, fontsize=8); ax.set_ylim(0, 95)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("F1 en campo (%)")
axes[1].legend(loc="upper right")
fig.suptitle("Efecto del slicing sobre el test de trabajo de campo",
             fontsize=9, fontweight="bold")
guardar(fig, "btm_sahi")
