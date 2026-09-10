#!/usr/bin/env python3
"""Muestra seis escenas del test fotointerpretado con las anotaciones superpuestas.

Recupera las anotaciones de los volcados de predicciones del bloque de tamaño y las dibuja sobre las
imágenes correspondientes, escogidas para cubrir ejemplares grandes y aislados, colonias densas,
mezclas de tamaños, fondos heterogéneos y formas irregulares. Escribe dataset_ejemplos.pdf, la
figura que ilustra el carácter polimórfico de los enebros.
"""
import sys, json
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from estilo import TEXTWIDTH, guardar

IMG_PI = "/home/santana/Documents/TFM/Photo_Interpretation_Data_jpg/Test/Images"
IMG_FW = "/home/santana/Documents/TFM/Field_Work_Data_jpg/External_Val_Data/Images"
SRV = "/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup"
EXP = f"{SRV}/experiments/juniperus_seg_by_size_20260830_202803"

VERDE = "#00E676"


def carga(split):
    j = json.load(open(f"{EXP}/predictions/{split}/dense_baseline.json"))
    return {im["image"]: im for im in j["images"]}


def dibuja(ax, nombre, datos, carpeta, titulo, gt=True):
    img = Image.open(f"{carpeta}/{nombre.replace('.tif', '.jpg')}")
    ax.imshow(np.asarray(img))
    if gt:
        for g in datos[nombre]["gt"]:
            ax.add_patch(MplPoly(np.array(g).reshape(-1, 2), closed=True,
                                 fill=False, edgecolor=VERDE, linewidth=0.9))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(titulo, fontsize=6.5, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)


pi, fw = carga("test"), carga("field_work")

CASOS = [
    ("Img_541.tif", pi, IMG_PI, "Ejemplares grandes y aislados"),
    ("Img_298.tif", pi, IMG_PI, "Colonia densa (54 individuos)"),
    ("Img_620.tif", pi, IMG_PI, "Mezcla de tamaños"),
    ("Img_335.tif", pi, IMG_PI, "Ejemplares pequeños dispersos"),
    ("Img_295.tif", pi, IMG_PI, "Fondo heterogéneo"),
    ("Img_542.tif", pi, IMG_PI, "Formas alargadas e irregulares"),
]

fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 3.6))
for ax, (n, d, c, t) in zip(axes.ravel(), CASOS):
    dibuja(ax, n, d, c, t)
fig.suptitle("El carácter polimórfico de los enebros en el conjunto de datos",
             fontsize=9, fontweight="bold")
fig.text(0.5, -0.01, "Anotaciones de los expertos en verde. Imágenes de 448$\\times$448 px "
                     "(13 cm/píxel) del conjunto de test fotointerpretado.",
         ha="center", fontsize=6)
guardar(fig, "dataset_ejemplos")
