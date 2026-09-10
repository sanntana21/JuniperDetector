#!/usr/bin/env python3
"""Sitúa cada mecanismo calibrado en el plano de F1 de los dos conjuntos de evaluación.

Toma de calibrado.py el mejor F1 vía S-IoU y su umbral de confianza óptimo para el
modelo único, los ensembles, el ensemble no especializado y el model merging de cada
bloque de experimentos, y los representa frente a la referencia. Escribe la figura
comparativa_global.pdf y detalla al pie los puntos que quedan fuera de los ejes.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
from estilo import TEXTWIDTH, C, guardar
import calibrado

REF = {"test": 87.87, "field_work": 76.86}
BLOQUES = [
    ("H", "juniperus_seg_homogeneous_20260905_105741",   False),
    ("E", "juniperus_seg_heterogeneous_20260905_111758", False),
    ("T", "juniperus_seg_by_size_20260830_202803",       False),
    ("R", "juniperus_seg_by_region_20260831_073029",     False),
    ("X", "legacy:mask2former_seg",                     True),
    ("C", "copypaste_test",                              True),
    ("D", "distilled",                                   True),
]
CLAVE = ("H, data augmentation homogénea.   E, data augmentation heterogénea.   "
         "T, tamaño del arbusto.   R, región geográfica.\n"
         "X, arquitectura transformer.   C, copy-paste.   D, destilación.")
MECS = [("dense_baseline",               "Modelo único",                      C["denso"],    "s"),
        ("btm_forest_ensemble_uniform",  "Ensemble de expertos",              C["ensemble"], "o"),
        ("btm_forest_ensemble_weighted", "Ensemble con pesos por fiabilidad", "#7D3C98",     "D"),
        ("btm_forest_soup",              "Model merging",                     C["soup"],     "v"),
        ("random_ensemble",              "Ensemble no especializado",         C["control"],  "^")]

XL, YL = (73, 92), (60, 86)
fig, ax = plt.subplots(figsize=(TEXTWIDTH, 4.6))
ax.add_patch(plt.Rectangle((REF["test"], REF["field_work"]), 100, 100,
                           color="#EAF2F8", zorder=0))
ax.axvline(REF["test"], color=C["referencia"], ls="--", lw=0.9, zorder=2)
ax.axhline(REF["field_work"], color=C["referencia"], ls="--", lw=0.9, zorder=2)
ax.plot(REF["test"], REF["field_work"], marker="*", ms=14, color=C["referencia"],
        zorder=8, ls="none")
ax.annotate("referencia", (REF["test"], REF["field_work"]), textcoords="offset points",
            xytext=(-6, -11), ha="right", fontsize=6.2, color=C["referencia"])
ax.text(91.6, 85.4, "mejor que la referencia\nen los dos conjuntos", fontsize=5.8,
        color="#4A6D80", va="top", ha="right")

vistas, fuera = set(), []
for nom, exp, unico in BLOQUES:
    mecs = [MECS[0]] if unico else MECS
    for mec, lab, col, mk in mecs:
        t = calibrado.optimo(exp, "test", mec)
        w = calibrado.optimo(exp, "field_work", mec)
        if t is None or w is None:
            continue
        if not (XL[0] <= t[0] <= XL[1] and YL[0] <= w[0] <= YL[1]):
            fuera.append((nom, lab, t[0], w[0]))
            continue
        ax.plot(t[0], w[0], marker=mk, ms=6.0, color=col, ls="none", zorder=5,
                mec="white", mew=0.8, label=lab if lab not in vistas else None)
        vistas.add(lab)
        ax.annotate(nom, (t[0], w[0]), textcoords="offset points", xytext=(5, 1.5),
                    fontsize=6.0, color=col, zorder=9, fontweight="bold")

ax.set_xlim(*XL); ax.set_ylim(*YL)
ax.set_xlabel("F1 sobre PI, el test fotointerpretado (%)")
ax.set_ylabel("F1 sobre FW, el test de trabajo de campo (%)")
ax.set_title("Cada mecanismo en su propio umbral de confianza óptimo")
ax.legend(loc="lower left", ncol=2, fontsize=6.0, handletextpad=0.3,
          borderaxespad=0.5, columnspacing=0.8)
pie = CLAVE
if fuera:
    pie += "\nQuedan fuera de los ejes por abajo:  " + ";  ".join(
        f"{n}, {l.lower()} ({t:.1f} / {w:.1f})".replace(".", ",") for n, l, t, w in fuera)
fig.text(0.0, -0.03, pie, fontsize=5.2, va="top", ha="left")
guardar(fig, "comparativa_global")
