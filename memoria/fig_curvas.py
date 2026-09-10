#!/usr/bin/env python3
"""Dibuja las curvas de F1 frente al umbral de confianza de cada mecanismo.

Obtiene las métricas por umbral a través del módulo calibrado y compone una rejilla con los cuatro
bloques que construyen expertos y los dos conjuntos de evaluación, marcando en cada curva el punto
que maximiza su F1. Escribe curvas_umbral.pdf, que justifica leer cada mecanismo calibrado en lugar
de al umbral por defecto de 0,25.
"""
import sys
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import matplotlib.pyplot as plt
from estilo import TEXTWIDTH, C, guardar
import calibrado

REF = {"test": 87.87, "field_work": 76.86}
BLOQUES = [
    ("juniperus_seg_homogeneous_20260905_105741",   "Data augmentation, homogénea"),
    ("juniperus_seg_heterogeneous_20260905_111758", "Data augmentation, heterogénea"),
    ("juniperus_seg_by_size_20260830_202803",       "Tamaño del arbusto"),
    ("juniperus_seg_by_region_20260831_073029",     "Región geográfica"),
]
MECS = [("dense_baseline",               "Modelo único",                    C["denso"],   "-"),
        ("btm_forest_ensemble_uniform",  "Ensemble de expertos",            C["ensemble"], "-"),
        ("btm_forest_ensemble_weighted", "Ensemble con pesos por fiabilidad", "#7D3C98",   "-."),
        ("btm_forest_soup",              "Model merging",                   C["soup"],    "--"),
        ("random_ensemble",              "Ensemble no especializado",       C["control"], ":")]

d = calibrado._carga()
fig, axes = plt.subplots(len(BLOQUES), 2, figsize=(TEXTWIDTH, 7.4),
                         sharex=True, sharey="row")
for i, (exp, titulo) in enumerate(BLOQUES):
    for j, (split, etiq) in enumerate((("test", "Test (PI)"), ("field_work", "Campo (FW)"))):
        ax = axes[i][j]
        ax.axhline(REF[split], color=C["referencia"], ls="--", lw=0.9, zorder=1)
        for mec, lab, col, st in MECS:
            g = d[(d.experiment == exp) & (d.split == split) &
                  (d.mechanism == mec)].sort_values("score_thr")
            if g.empty:
                continue
            ax.plot(g.score_thr, g.f1, st, color=col, lw=1.2, zorder=3,
                    label=lab if (i == 0 and j == 0) else None)
            k = g.f1.idxmax()
            ax.plot(g.loc[k, "score_thr"], g.loc[k, "f1"], "o", ms=4.0, color=col,
                    mec="white", mew=0.7, zorder=5)
        ax.set_ylim(30, 95)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(etiq, fontsize=8)
        if j == 0:
            ax.set_ylabel(titulo, fontsize=6.2)
        if i == len(BLOQUES) - 1:
            ax.set_xlabel("Umbral de confianza")
        ax.tick_params(labelsize=6.5)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.055))
fig.suptitle("F1 frente al umbral de confianza, por bloque y conjunto",
             fontsize=9, fontweight="bold")
guardar(fig, "curvas_umbral")
