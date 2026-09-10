#!/usr/bin/env python3
"""Genera la misma imagen bajo los tres regímenes de data augmentation que definen los expertos.

Aplica a una imagen del conjunto de entrenamiento una realización representativa de
los parámetros reales de escala, iluminación y geometría de cada régimen. Escribe la
figura regimenes.pdf, que ilustra el criterio de especialización por divergencia
inducida.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from estilo import TEXTWIDTH, guardar

BASE = Path("/home/santana/Documents/TFM/dataset_yolo_seg/images/train")
NOMBRE = "Img_472"
GRIS = (114, 114, 114)          # relleno por defecto de Ultralytics
img = cv2.imread(str(BASE / f"{NOMBRE}.tif"))
H, W = img.shape[:2]


def escala(im, f=1.45):
    """E1: reescalado, parametro scale=0.9 (rango +-90 %)."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, f)
    return cv2.warpAffine(im, M, (W, H), borderValue=GRIS)


def iluminacion(im, dh=6, fs=0.35, fv=1.55):
    """E2: tono, saturacion y brillo, parametros hsv 0.03 / 0.9 / 0.7."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * fs, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * fv, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def geometria(im, deg=22.0, shear=8.0):
    """E3: rotacion y cizalladura, parametros degrees=25 y shear=10."""
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    im = cv2.warpAffine(im, M, (W, H), borderValue=GRIS)
    t = np.tan(np.deg2rad(shear))
    S = np.float32([[1, t, -t * H / 2], [0, 1, 0]])
    return cv2.warpAffine(im, S, (W, H), borderValue=GRIS)


PANELES = [
    (escala(img), "E1 · escala\nscale = 0,9"),
    (iluminacion(img), "E2 · iluminación\nhsv = 0,03 / 0,9 / 0,7"),
    (geometria(img), "E3 · geometría\ndegrees = 25, shear = 10"),
]

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15))
for ax, (im, tit) in zip(axes, PANELES):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(tit, fontsize=6.3, pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.suptitle("Lo que ve cada experto durante el entrenamiento",
             fontsize=9, fontweight="bold")
guardar(fig, "regimenes")
