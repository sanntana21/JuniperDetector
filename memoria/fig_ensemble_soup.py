#!/usr/bin/env python3
"""Dibuja el esquema que contrapone el ensemble y el model soup.

Compone un diagrama sin datos con los dos flujos: el ensemble ejecuta los tres expertos y funde sus
predicciones con Weighted Box Fusion, mientras que el model soup promedia sus parámetros antes de
ver ninguna imagen y deja un único modelo. Escribe ensemble_soup.pdf, la figura que fija la
distinción entre combinar predicciones y combinar parámetros.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from estilo import TEXTWIDTH, C, guardar

NAR, ROSA, GRIS = C["ensemble"], C["soup"], "#8C8C8C"

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 106)
ax.axis("off")


def caja(x, y, w, h, texto, color, fs=5.8, negrita=False, relleno="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", linewidth=1.1,
                                edgecolor=color, facecolor=relleno, zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            zorder=3, linespacing=1.3, fontweight="bold" if negrita else "normal")


def fl(x1, y1, x2, y2, color=GRIS):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                 linewidth=1.0, color=color, zorder=1, shrinkA=1, shrinkB=2))


ax.plot([50, 50], [4, 94], color="#DDDDDD", lw=0.9, zorder=0)
ax.text(24, 97, "Ensemble", ha="center", fontsize=8.4, fontweight="bold", color=NAR)
ax.text(76, 97, "Model soup", ha="center", fontsize=8.4, fontweight="bold", color=ROSA)
ax.text(24, 92.5, "se combinan las salidas", ha="center", fontsize=6, style="italic", color="#555555")
ax.text(76, 92.5, "se combinan los pesos", ha="center", fontsize=6, style="italic", color="#555555")

# ---------------------------------------------------------------- ensemble
caja(14, 79, 20, 7, "Imagen de entrada", GRIS)
for i, x in enumerate((1, 17, 33)):
    fl(24, 79, x + 6.5, 72.5)
    caja(x, 65, 13, 7.5, f"Experto {i+1}", NAR)
    fl(x + 6.5, 65, x + 6.5, 58.5)
    caja(x, 51, 13, 7.5, "Predicciones", GRIS, fs=5.4)
    fl(x + 6.5, 51, 24, 44.5)
caja(6, 37, 36, 7.5, "Weighted Box Fusion", NAR, negrita=True)
fl(24, 37, 24, 30.5)
caja(9, 23, 30, 7.5, "Predicciones finales", GRIS)
ax.text(24, 14, "Los tres modelos siguen existiendo.\nEn producción hay que ejecutar los tres.",
        ha="center", va="center", fontsize=5.8, linespacing=1.4)

# ---------------------------------------------------------------- model soup
for i, x in enumerate((53, 69, 85)):
    caja(x, 79, 13, 7.5, f"Experto {i+1}", ROSA)
    fl(x + 6.5, 79, 76, 72.5)
caja(58, 65, 36, 7.5, "Promedio de sus parámetros", ROSA, negrita=True)
fl(76, 65, 76, 58.5)
caja(64, 51, 24, 7.5, "Un solo modelo", ROSA)
caja(50, 51, 11, 7.5, "Imagen", GRIS, fs=5.4)
fl(61, 54.75, 63.5, 54.75)
fl(76, 51, 76, 44.5)
caja(61, 37, 30, 7.5, "Predicciones finales", GRIS)
ax.text(76, 27, "Los expertos se funden antes de ver\nninguna imagen. En producción queda\nun único modelo.",
        ha="center", va="center", fontsize=5.8, linespacing=1.4)
ax.text(76, 15, "Exige que los tres expertos tengan la misma\narquitectura y el mismo punto de partida.",
        ha="center", va="center", fontsize=5.6, style="italic", color="#555555", linespacing=1.4)

fig.suptitle("Las dos formas de combinar varios modelos", fontsize=9.5, fontweight="bold", y=0.995)
guardar(fig, "ensemble_soup")
