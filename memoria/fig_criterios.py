#!/usr/bin/env python3
"""Ilustra qué ve cada experto según el criterio de especialización.

Toma imágenes y anotaciones reales de dataset_yolo_seg y compone una rejilla de tres filas: el
régimen de data augmentation, que altera la imagen y conserva todas las anotaciones; el tamaño del
arbusto, que reparte las anotaciones; y la región geográfica, que reparte las imágenes. Escribe
criterios.pdf, la figura que sostiene la explicación de los tres repartos del problema.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from pathlib import Path
from estilo import TEXTWIDTH, C, guardar

BASE = Path("/home/santana/Documents/TFM/dataset_yolo_seg")
GRIS_RELLENO = (114, 114, 114)          # relleno por defecto de Ultralytics
VERDE, APAGADO = "#00E676", "#9E9E9E"
COL_TAM = [C["experto1"], C["experto2"], C["soup"]]     # pequeno, mediano, grande
UMBRAL = (197.8, 1156.2)                # los mismos que definen los subconjuntos

ESCENA = "Img_458"                      # mezcla clara de los tres tamanos
# una imagen de cada region, del reparto geografico del bloque de region
REGION = ["Img_666", "Img_312"]   # extremo oeste y extremo este del reparto


def carga(nombre, split="train"):
    img = cv2.cvtColor(cv2.imread(str(BASE / f"images/{split}/{nombre}.tif")),
                       cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    polis = []
    for ln in (BASE / f"labels/{split}/{nombre}.txt").read_text().splitlines():
        p = ln.split()
        if len(p) < 7:
            continue
        c = np.array(p[1:], float).reshape(-1, 2) * [w, h]
        area = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], 1)) -
                         np.dot(c[:, 1], np.roll(c[:, 0], 1)))
        rango = 0 if area <= UMBRAL[0] else (1 if area <= UMBRAL[1] else 2)
        polis.append((c, rango))
    return img, polis


# --- las tres transformaciones, con los parametros reales de cada experto ---
def _aplica(im, polis, M):
    """Deforma imagen y anotaciones con la misma matriz afin."""
    h, w = im.shape[:2]
    out = cv2.warpAffine(im, M, (w, h), borderValue=GRIS_RELLENO)
    mov = [(np.c_[c, np.ones(len(c))] @ M.T, t) for c, t in polis]
    return out, mov


def escala(im, polis, f=1.45):
    h, w = im.shape[:2]
    return _aplica(im, polis, cv2.getRotationMatrix2D((w / 2, h / 2), 0, f))


def iluminacion(im, polis, dh=6, fs=0.35, fv=1.55):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * fs, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * fv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), polis


def geometria(im, polis, deg=22.0, shear=8.0):
    h, w = im.shape[:2]
    im, polis = _aplica(im, polis, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0))
    t = np.tan(np.deg2rad(shear))
    return _aplica(im, polis, np.float32([[1, t, -t * h / 2], [0, 1, 0]]))


img, polis = carga(ESCENA)
fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, 5.0))


def panel(ax, imagen, dibuja, titulo):
    ax.imshow(imagen)
    for coords, color in dibuja:
        fuera = color == APAGADO
        ax.add_patch(MplPoly(coords, closed=True, fill=False, edgecolor=color,
                             linewidth=0.6 if fuera else 1.4,
                             linestyle=":" if fuera else "-", alpha=0.8 if fuera else 1.0))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(titulo, fontsize=6.0, pad=2.5)


# fila 1, regimen de data augmentation: misma imagen, mismas anotaciones
for ax, (trans, etq) in zip(axes[0], [(escala, "E1 · escala"),
                                      (iluminacion, "E2 · iluminación"),
                                      (geometria, "E3 · geometría")]):
    im_t, pol_t = trans(img, polis)
    panel(ax, im_t, [(c, VERDE) for c, _ in pol_t], etq)

# fila 2, tamano: misma imagen, cambia que anotaciones estan marcadas
for ax, (r, etq) in zip(axes[1], enumerate(["Experto de pequeños",
                                            "Experto de medianos",
                                            "Experto de grandes"])):
    panel(ax, img, [(c, COL_TAM[r] if t == r else APAGADO) for c, t in polis], etq)

# fila 3, region: imagenes distintas, todas sus anotaciones
for k, ax in enumerate(axes[2]):
    if k >= len(REGION):
        ax.axis("off")
        continue
    im_r, pol_r = carga(REGION[k])
    panel(ax, im_r, [(c, VERDE) for c, _ in pol_r], f"Experto de la región {k}")

for fila, etq in enumerate(["Régimen de\n\\textit{data augmentation}".replace("\\textit{", "").replace("}", ""),
                            "Tamaño\ndel arbusto", "Región\ngeográfica"]):
    axes[fila][0].set_ylabel(etq, fontsize=7)

fig.suptitle("Qué ve cada experto según el criterio de especialización",
             fontsize=9, fontweight="bold")
guardar(fig, "criterios")
