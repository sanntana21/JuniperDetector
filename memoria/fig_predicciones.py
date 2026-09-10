#!/usr/bin/env python3
"""Contrapone predicción y anotación sobre seis escenas de acierto y de fallo.

Lee las predicciones del modelo único del experimento de especialización por tamaño
sobre el test fotointerpretado y el test de trabajo de campo, filtra por confianza
0,25 y superpone en cada panel la anotación del experto y la predicción. Escribe la
figura predicciones_ejemplos.pdf, que documenta los modos de fallo del modelo.
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
SRV = "/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup"
EXP = f"{SRV}/experiments/juniperus_seg_by_size_20260830_202803"
CONF = 0.25
VERDE, NARANJA = "#00E676", "#FF6D00"

DATOS = {sp: {im["image"]: im for im in
              json.load(open(f"{EXP}/predictions/{sp}/dense_baseline.json"))["images"]}
         for sp in IMG}


def panel(ax, split, nombre, titulo):
    d = DATOS[split][nombre]
    ax.imshow(np.asarray(Image.open(f"{IMG[split]}/{nombre.replace('.tif', '.jpg')}")))
    for g in d["gt"]:
        ax.add_patch(MplPoly(np.array(g).reshape(-1, 2), closed=True, fill=False,
                             edgecolor=VERDE, linewidth=1.0))
    for p, s in zip(d["polys"], d["scores"]):
        if s >= CONF:
            ax.add_patch(MplPoly(np.array(p).reshape(-1, 2), closed=True, fill=False,
                                 edgecolor=NARANJA, linewidth=1.0, linestyle="--"))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(titulo, fontsize=7, pad=3)
    for s_ in ax.spines.values():
        s_.set_visible(False)


CASOS = [
    ("test", "Img_220.tif", "(a) Acierto limpio"),
    ("test", "Img_298.tif", "(b) Fragmentación"),
    ("field_work", "Img_111.tif", "(c) Fusión de vecinos"),
    ("field_work", "Img_81.tif", "(d) Pequeños no detectados"),
    ("test", "Img_201.tif", "(e) Falsos positivos"),
    ("field_work", "Img_34.tif", "(f) Fondo texturizado"),
]

fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.7))
for ax, (sp, n, t) in zip(axes.ravel(), CASOS):
    panel(ax, sp, n, t)
fig.suptitle("Cómo acierta y cómo falla el modelo", fontsize=9, fontweight="bold")
fig.legend(handles=[Line2D([], [], color=VERDE, lw=1.4, label="Anotación del experto"),
                    Line2D([], [], color=NARANJA, lw=1.4, ls="--", label="Predicción del modelo")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.055), fontsize=6.5)
guardar(fig, "predicciones_ejemplos")
