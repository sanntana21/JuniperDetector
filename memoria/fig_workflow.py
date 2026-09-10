#!/usr/bin/env python3
"""Dibuja el esquema del flujo de trabajo seguido en el proyecto.

Compone un diagrama de tres bandas con el modelo base de referencia, las estrategias
exploradas (arquitecturas transformer, ensemble, model merging, inferencia sobre
recortes y generación de datos) y los conjuntos y métricas con que se miden. Escribe
la figura workflow.pdf, que sirve de mapa de la metodología.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from estilo import TEXTWIDTH, C, guardar

GRIS, GRISC = "#8C8C8C", "#F2F2F2"

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 5.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 110)
ax.axis("off")


def banda(y, h, titulo):
    ax.add_patch(Rectangle((0, y), 100, h, facecolor=GRISC, edgecolor="none", zorder=0))
    ax.text(1.5, y + h - 3.0, titulo, fontsize=7.2, fontweight="bold", va="top", ha="left")


def caja(x, y, w, h, tit, cuerpo, color, fs=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7",
                                linewidth=1.1, edgecolor=color, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h - 2.6, tit, ha="center", va="top", fontsize=fs + 0.5,
            fontweight="bold", color=color, zorder=3, linespacing=1.25)
    ax.text(x + w / 2, y + h - 8.2, cuerpo, ha="center", va="top", fontsize=fs,
            zorder=3, linespacing=1.35)


def flecha(x, y1, y2):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.3, color=GRIS, zorder=1))


# ---------------------------------------------------- 1. modelo base
banda(87, 23, "1. Modelo base")
caja(20, 89, 60, 14,
     "Mask R-CNN con backbone ResNet-101",
     "Khaldi et al. (2024)\nSegmentación de instancias, una sola clase",
     C["referencia"], fs=6.0)
flecha(50, 86, 82.5)

# ---------------------------------------------------- 2. estrategias
banda(31, 50, "2. Estrategias exploradas")
caja(2, 57, 30, 17, "Arquitecturas\nbasadas en Transformers",
     "DETR, Deformable DETR,\nCo-DETR, Mask2Former", C["denso"])
caja(35, 57, 30, 17, "Ensemble de modelos",
     "Expertos entrenados por separado\ny fusión de sus predicciones\nmediante Weighted Box Fusion",
     C["ensemble"])
caja(68, 57, 30, 17, "Model merging",
     "Promediado de los parámetros\nde los expertos en un solo\nmodelo (soup, TIES-merging)",
     C["soup"])
ax.text(50, 53.5, "Criterios de especialización de los expertos: régimen de data augmentation,\n"
                "región geográfica y tamaño del arbusto",
        ha="center", va="center", fontsize=5.6, style="italic", color="#444444")
caja(2, 34, 46, 15, "Inferencia sobre recortes",
     "La imagen se divide en recortes solapados\ny se evalúa cada uno por separado", C["experto1"])
caja(52, 34, 46, 15, "Generación de datos de entrenamiento",
     "Copiado y pegado de instancias reales\ny destilación del ensemble en un modelo",
     C["control"])
flecha(50, 30, 26.5)

# ---------------------------------------------------- 3. evaluacion
banda(0, 27, "3. Evaluación y comparación")
caja(2, 8, 30, 13, "Entrenamiento",
     "Entrenamiento, 570 img / 5.459\nValidación, 67 img / 660", C["gris"])
caja(35, 8, 30, 13, "Test",
     "Fotointerpretado, 75 img / 690\nTrabajo de campo, 124 img / 1.771", C["gris"])
caja(68, 8, 30, 13, "Métricas",
     "F1 con IoU y S-IoU\na 50 \\% y 75 \\% de solapamiento".replace("\\%", "%"), C["gris"])
ax.text(50, 4.0, "Cada estrategia se compara con el modelo base y con dos referencias internas, "
                 "un modelo único\ny un ensemble del mismo número de modelos entrenados sin especializar",
        ha="center", va="center", fontsize=5.8, linespacing=1.4)

fig.suptitle("Flujo de trabajo", fontsize=9.5, fontweight="bold", y=0.985)
guardar(fig, "workflow")
