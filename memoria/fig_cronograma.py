#!/usr/bin/env python3
"""Dibuja el cronograma del proyecto, de octubre de 2025 a septiembre de 2026.

Parte de las fases declaradas en el propio script y las representa como barras sobre un eje
temporal, con las dos mitades del trabajo marcadas como bandas de fondo. Escribe cronograma.pdf,
que sitúa en el tiempo la exploración de arquitecturas, el bloque de expertos y la redacción de la
memoria.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from estilo import TEXTWIDTH, C, guardar

# (etiqueta, inicio, fin, color)
FASES = [
    ("Estudio del trabajo previo\ny puesta en marcha",
     date(2025, 10, 1), date(2025, 11, 20), C["gris"]),
    ("Exploración de arquitecturas\nDETR, Deformable DETR, Co-DETR",
     date(2025, 11, 20), date(2026, 4, 20), C["denso"]),
    ("Mask2Former y ensayos finales\nde la familia DETR",
     date(2026, 4, 20), date(2026, 6, 10), C["denso"]),
    ("Reformulación del problema\ny diseño del segundo bloque",
     date(2026, 6, 10), date(2026, 7, 5), C["gris"]),
    ("Expertos por régimen\nde \\textit{data augmentation}",
     date(2026, 7, 5), date(2026, 7, 26), C["ensemble"]),
    ("Expertos por tamaño\ny por región geográfica",
     date(2026, 7, 20), date(2026, 8, 20), C["ensemble"]),
    ("Inferencia sobre recortes\ny calibración del umbral",
     date(2026, 8, 10), date(2026, 8, 31), C["experto1"]),
    ("Ampliación de la señal\ncopy-paste y destilación",
     date(2026, 8, 18), date(2026, 9, 3), C["control"]),
    ("Análisis de resultados",
     date(2026, 6, 1), date(2026, 9, 8), C["soup"]),
    ("Redacción de la memoria",
     date(2026, 3, 1), date(2026, 9, 12), C["soup"]),
]

# Las dos mitades del proyecto, como banda de fondo
BLOQUES = [("Primera fase", date(2025, 10, 1), date(2026, 6, 10), "#F4F7FA"),
           ("Segunda fase", date(2026, 6, 10), date(2026, 9, 12), "#FBF5F0")]

fig, ax = plt.subplots(figsize=(TEXTWIDTH, 3.2))

for etq, ini, fin, col in BLOQUES:
    ax.axvspan(ini, fin, color=col, zorder=0)
    ax.text(mdates.date2num(ini) + (mdates.date2num(fin) - mdates.date2num(ini)) / 2,
            len(FASES) + 0.55, etq, ha="center", va="center", fontsize=6.2,
            color="#666666", style="italic")

for i, (etq, ini, fin, col) in enumerate(FASES):
    y = len(FASES) - i
    ax.barh(y, (fin - ini).days, left=ini, height=0.58, color=col, alpha=0.92, zorder=3)
    ax.text(mdates.date2num(ini) - 8, y, etq.replace("\\textit{", "").replace("}", ""),
            ha="right", va="center", fontsize=5.9, linespacing=1.25)

ax.set_ylim(0.3, len(FASES) + 0.95)
ax.set_yticks([])
ax.set_xlim(date(2025, 6, 15), date(2026, 10, 5))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]
ax.xaxis.set_major_formatter(plt.FuncFormatter(
    lambda v, _: f"{MESES[mdates.num2date(v).month - 1]}\n{mdates.num2date(v).year}"))
ax.tick_params(axis="x", labelsize=6)
ax.grid(axis="x", zorder=1)
ax.spines["left"].set_visible(False)
fig.suptitle("Cronograma del proyecto", fontsize=9, fontweight="bold")
guardar(fig, "cronograma")
