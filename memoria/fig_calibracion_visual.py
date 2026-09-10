#!/usr/bin/env python3
"""Dibuja las predicciones del model merging por tamaño a dos umbrales de confianza.

Lee el volcado de predicciones del model merging del bloque de tamaño sobre una escena del test
fotointerpretado y otra del test de trabajo de campo, y superpone las detecciones que superan el
umbral común de 0,25 junto a las que solo aparecen al bajarlo a 0,02. Escribe
calibracion_visual.pdf, que ilustra sobre la imagen por qué cada mecanismo debe leerse calibrado.
"""
import sys, json
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.lines import Line2D
from PIL import Image
from estilo import TEXTWIDTH, guardar

IMG = {"test": "/home/santana/Documents/TFM/Photo_Interpretation_Data_jpg/Test/Images",
       "field_work": "/home/santana/Documents/TFM/Field_Work_Data_jpg/External_Val_Data/Images"}
EXP = ("/home/santana/Documents/docs_TFM/analisis/exp8/experiments/"
       "juniperus_seg_by_size_20260830_163740")
COMUN, BAJO = 0.25, 0.02
VERDE, NARANJA, MORADO = "#00E676", "#FF6D00", "#7B1FA2"

DATOS = {sp: {im["image"]: im for im in
              json.load(open(f"{EXP}/predictions/{sp}/btm_forest_soup.json"))["images"]}
         for sp in IMG}

# Dos escenas donde el efecto es nitido, una por conjunto de evaluacion.
CASOS = [("test", "Img_513.tif", "Test (PI)"),
         ("field_work", "Img_76.tif", "Campo (FW)")]


def panel(ax, split, nombre, umbral, mostrar_bajas):
    d = DATOS[split][nombre]
    ax.imshow(np.asarray(Image.open(f"{IMG[split]}/{nombre.replace('.tif', '.jpg')}")))
    for g in d["gt"]:
        ax.add_patch(MplPoly(np.array(g).reshape(-1, 2), closed=True, fill=False,
                             edgecolor=VERDE, linewidth=0.9))
    n_alta = n_baja = 0
    for p, s in zip(d["polys"], d["scores"]):
        if s >= COMUN:
            col, n_alta = NARANJA, n_alta + 1
        elif mostrar_bajas and s >= BAJO:
            col, n_baja = MORADO, n_baja + 1
        else:
            continue
        ax.add_patch(MplPoly(np.array(p).reshape(-1, 2), closed=True, fill=False,
                             edgecolor=col, linewidth=0.9, linestyle="--"))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    total = n_alta + n_baja
    ax.set_title(f"umbral {umbral:.2f}".replace(".", ",") +
                 f" · {total} detecci{'ones' if total != 1 else 'ón'}"
                 f" · {len(d['gt'])} anotadas", fontsize=6.4, pad=2.5)


fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, 4.5))
for fila, (sp, nombre, titulo) in enumerate(CASOS):
    panel(axes[fila][0], sp, nombre, COMUN, False)
    panel(axes[fila][1], sp, nombre, BAJO, True)
    axes[fila][0].set_ylabel(titulo, fontsize=7)

fig.suptitle("Predicciones del model merging por tamaño a dos umbrales",
             fontsize=9, fontweight="bold")
fig.legend(handles=[Line2D([], [], color=VERDE, lw=1.4, label="Anotación del experto"),
                    Line2D([], [], color=NARANJA, lw=1.4, ls="--",
                           label="Detección con confianza $\\geq$ 0,25"),
                    Line2D([], [], color=MORADO, lw=1.4, ls="--",
                           label="Detección con confianza entre 0,02 y 0,25")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.085), fontsize=6.4)
guardar(fig, "calibracion_visual")
