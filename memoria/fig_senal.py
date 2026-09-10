#!/usr/bin/env python3
"""Dibuja el esquema de las dos vías empleadas para generar más señal de entrenamiento.

Contrapone en un diagrama el aumento por copy-paste, que añade recortes reales sobre
la anotación humana, y la destilación del ensemble de expertos por tamaño, que la
sustituye por pseudo-labels, e indica que la validación conserva en ambos casos la
anotación original. Escribe la figura senal_vias.pdf.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from estilo import TEXTWIDTH, C, guardar

VERDE, ROJO, GRIS = C["control"], C["ensemble"], C["gris"]

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.15))
ax.set_xlim(0, 100); ax.set_ylim(0, 66)
ax.axis("off")


def caja(x, y, w, h, texto, color, fs=6.0, negrita=False, relleno="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                linewidth=1.1, edgecolor=color, facecolor=relleno,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
            zorder=3, fontweight="bold" if negrita else "normal", linespacing=1.35)


def flecha(x1, y1, x2, y2, color=GRIS):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.0, color=color,
                                 zorder=1, shrinkA=1, shrinkB=2))


ax.text(23, 62, "Aumento por copiar y pegar", ha="center", fontsize=7.4,
        fontweight="bold", color=VERDE)
ax.text(77, 62, "Destilación del ensemble", ha="center", fontsize=7.4,
        fontweight="bold", color=ROJO)
ax.plot([50, 50], [3, 59], color="#DDDDDD", lw=0.9, zorder=0)

# --- via 1
caja(2, 47, 42, 9, "570 imágenes\n5.459 etiquetas humanas", VERDE)
flecha(23, 47, 23, 41)
caja(2, 30, 42, 10, "Se pegan de 1 a 3 recortes reales\ndel rango pequeño en cada imagen", VERDE)
flecha(23, 30, 23, 24)
caja(2, 14, 42, 9, "Etiquetas humanas\nmás las añadidas", VERDE, negrita=True)

# --- via 2
caja(56, 47, 42, 9, "Los 3 expertos por tamaño\nactúan como profesor", ROJO)
flecha(77, 47, 77, 41)
caja(56, 30, 42, 10, "Generan pseudo-etiquetas sobre\nlas mismas 570 imágenes", ROJO)
flecha(77, 30, 77, 24)
caja(56, 14, 42, 9, "Solo pseudo-etiquetas\nlas humanas se descartan", ROJO, negrita=True)

flecha(23, 14, 23, 9); flecha(77, 14, 77, 9)
caja(14, 1, 72, 8, "En los dos casos, la validación conserva las etiquetas humanas originales",
     GRIS, fs=6.2)

fig.suptitle("Añadir señal frente a sustituirla", fontsize=9, fontweight="bold")
guardar(fig, "senal_vias")
