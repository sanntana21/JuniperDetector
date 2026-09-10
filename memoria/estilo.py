"""Define el estilo gráfico común a todas las figuras de la memoria.

Biblioteca que importan el resto de los scripts de figuras: fija el ancho de texto del documento,
la paleta segura para daltonismo y los parámetros de dibujo de matplotlib, y ofrece la función
guardar, que escribe cada figura en PDF vectorial dentro de latex/imagenes/figuras. Centralizar
aquí el estilo mantiene homogéneas todas las figuras del trabajo.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path("/home/santana/Documents/docs_TFM/latex/imagenes/figuras")
FIGS.mkdir(parents=True, exist_ok=True)

# Ancho de texto del documento (11pt, a4paper, book) ~= 4.93 pulgadas
TEXTWIDTH = 4.93

# Paleta segura para daltonismo (Okabe-Ito) + grises de apoyo
C = {
    "referencia": "#000000",
    "denso":      "#0072B2",
    "ensemble":   "#D55E00",
    "soup":       "#CC79A7",
    "control":    "#009E73",
    "experto1":   "#E69F00",
    "experto2":   "#56B4E9",
    "experto3":   "#F0E442",
    "gris":       "#8C8C8C",
    "grisclaro":  "#DDDDDD",
}
CICLO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C8C8C", "#F0E442"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.titleweight": "bold",
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def guardar(fig, nombre):
    """Guarda la figura en PDF (vectorial) dentro de latex/imagenes/figuras."""
    out = FIGS / f"{nombre}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("  ->", out.name)
    return out


def anota_barras(ax, barras, fmt="{:.1f}", dx=0.0, dy=0.6, size=6):
    for b in barras:
        w = b.get_width()
        ax.text(w + dx, b.get_y() + b.get_height() / 2 + 0.0, fmt.format(w),
                va="center", ha="left", fontsize=size)
