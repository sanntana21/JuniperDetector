#!/usr/bin/env python3
"""Dibuja el F1 de cada mecanismo por rango de tamaño del arbusto anotado.

Lee por_bucket.csv y compone dos mapas de calor, uno por conjunto de evaluación, con los tres
expertos por tamaño, el ensemble de expertos, el ensemble no especializado y el modelo único,
medidos con F1 vía S-IoU. Escribe btm_diagonal.pdf, donde se aprecia que cada experto rinde mejor
en su propio rango.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from estilo import TEXTWIDTH, guardar

d = pd.read_csv("/home/santana/Documents/docs_TFM/analisis/por_bucket.csv")
FILAS = [("expert_E_small_solo", "Experto pequeño"),
         ("expert_E_medium_solo", "Experto mediano"),
         ("expert_E_large_solo", "Experto grande"),
         ("btm_forest_ensemble_uniform", "Ensemble de expertos"),
         ("random_ensemble", "Ensemble no especializado"),
         ("dense_baseline", "Modelo único")]
COLS = [("small", "Pequeños"), ("medium", "Medianos"), ("large", "Grandes")]

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
for ax, split, tit in zip(axes, ["test", "field_work"], ["Test (PI)", "Campo (FW)"]):
    M = np.array([[float(d[(d.split == split) & (d.metrica == "S-IoU") &
                           (d.variante == v) & (d.bucket == b)].f1.iloc[0])
                   for b, _ in COLS] for v, _ in FILAS])
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 55 else "black",
                    fontweight="bold" if i < 3 and M[i, j] == M[i].max() else "normal")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c for _, c in COLS], fontsize=7)
    ax.set_yticks(range(len(FILAS)))
    ax.set_yticklabels([n for _, n in FILAS] if split == "test" else [""] * len(FILAS),
                       fontsize=7)
    ax.set_title(tit, fontsize=8)
    ax.grid(False)
    ax.axhline(2.5, color="white", lw=2)
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("F1 vía S-IoU (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
fig.suptitle("F1 por rango de tamaño del arbusto anotado",
             fontsize=9, fontweight="bold")
guardar(fig, "btm_diagonal")
