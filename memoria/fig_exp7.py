#!/usr/bin/env python3
"""Dibuja el barrido del umbral de confianza de los dos mecanismos que generan más señal.

Lee exp7_barrido.csv y traza, para el test fotointerpretado y el test de trabajo
de campo, la curva de F1 vía S-IoU de copy-paste y de destilación frente al umbral
de confianza, marcando el máximo de cada una junto a las líneas de la referencia y
del modelo único calibrado. Escribe la figura exp7_barrido.pdf.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import pandas as pd
import matplotlib.pyplot as plt
from estilo import TEXTWIDTH, C, guardar
import calibrado as _cal

A = "/home/santana/Documents/docs_TFM/analisis"
d = pd.read_csv(f"{A}/exp7_barrido.csv")
r = pd.read_csv(f"{A}/reeval_conf025.csv")
EXP_DENSO = "juniperus_seg_by_size_20260830_202803"
PAPER = {"test": 87.87, "field_work": 76.86}

MECS = [("Copy-paste", "copy-paste", C["ensemble"]),
        ("Destilación", "destilado", C["soup"])]

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.7), sharey=True)
for ax, split, tit in zip(axes, ["test", "field_work"], ["Test (PI)", "Campo (FW)"]):
    for lab, m, col in MECS:
        s = d[(d.split == split) & (d.mechanism == m) & (d.metric == "S-IoU") &
              (d.overlap == 50)].sort_values("score_thr")
        ax.plot(s.score_thr, s.f1, color=col, marker="o", ms=2.6, label=lab)
        k = s["f1"].idxmax()
        ax.scatter([s.loc[k, "score_thr"]], [s.loc[k, "f1"]], color=col, s=34,
                   zorder=5, edgecolor="white", linewidth=0.7)
        ax.annotate(f"{s.loc[k,'f1']:.1f}".replace(".", ","), (s.loc[k, "score_thr"], s.loc[k, "f1"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=6, color=col, fontweight="bold")
    ax.axhline(PAPER[split], color="k", ls="--", lw=1.1)
    ax.annotate(f"referencia {PAPER[split]:.1f}".replace(".", ","), (0.72, PAPER[split]),
                textcoords="offset points", xytext=(0, 4), fontsize=6, ha="right")
    dn = _cal.optimo(EXP_DENSO, split, "dense_baseline")
    if dn:
        ax.axhline(dn[0], color=C["denso"], ls=":", lw=1.1)
        x0 = float(s["score_thr"].min())
        ax.annotate(f"modelo único {dn[0]:.1f}".replace(".", ","),
                    (x0, dn[0]), textcoords="offset points",
                    xytext=(2, -9), fontsize=6, ha="left", color=C["denso"])
    ax.set_xlabel("Umbral de confianza")
    ax.set_title(tit, fontsize=8)
    ax.set_ylim(50, 95)
axes[0].set_ylabel("F1 vía S-IoU (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.11), fontsize=6.5)
fig.suptitle("Barrido del umbral de confianza de los dos mecanismos de señal",
             fontsize=9, fontweight="bold")
guardar(fig, "exp7_barrido")
