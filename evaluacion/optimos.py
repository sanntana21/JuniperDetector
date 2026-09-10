#!/usr/bin/env python3
"""Localiza el umbral de confianza óptimo de cada mecanismo a partir del barrido hacia arriba.

Lee barrido_arriba.csv, se queda con S-IoU al 50 % de solapamiento y busca para cada
ejecución, conjunto y mecanismo el umbral que maximiza su F1, distinguiendo si la curva
gira dentro del rango explorado o si el máximo cae en el suelo de exportación y es solo una
cota inferior. Imprime la comparación por conjunto y escribe optimos.csv.
"""
import pandas as pd
from pathlib import Path

A = Path("/home/santana/Documents/docs_TFM/analisis")
REF = {"test": 87.87, "field_work": 76.86}
NOMBRE = {
    "juniperus_seg_homogeneous_20260905_105741":   "DA homogénea",
    "juniperus_seg_heterogeneous_20260905_111758": "DA heterogénea",
    "juniperus_seg_by_size_20260830_202803":       "Tamaño, natural 100 ép.",
    "juniperus_seg_by_size_20260830_163740":       "Tamaño, natural 50 ép.",
    "juniperus_seg_by_size_20260830_164608":       "Tamaño, cant. igualada 50",
    "juniperus_seg_by_size_20260830_221329":       "Tamaño, modelo grande 100",
    "juniperus_seg_by_region_20260831_073029":     "Región canónica",
    "juniperus_seg_by_region_20260828_114011":     "Región 1a ejecución",
    "juniperus_seg_by_size_20260828_072323":       "Entrenamiento conjunto",
}
MEC = {"dense_baseline": "Modelo único",
       "btm_forest_ensemble_uniform": "Ensemble de expertos",
       "btm_forest_soup": "Model merging",
       "random_ensemble": "Ensemble no especializado",
       "btm_forest_ensemble_weighted": "Ens. pesos por fiabilidad"}


def tabla():
    d = pd.read_csv(A / "barrido_arriba.csv")
    d = d[(d.metric == "S-IoU") & (d.overlap == 50)]
    filas = []
    for (exp, split, mec), g in d.groupby(["experiment", "split", "mechanism"]):
        g = g.sort_values("score_thr")
        i = g.f1.idxmax()
        thr_opt, f1_opt = g.loc[i, "score_thr"], g.loc[i, "f1"]
        suelo = g.score_thr.min()
        f1_025 = g[g.score_thr == 0.25].f1
        f1_025 = float(f1_025.iloc[0]) if len(f1_025) else float("nan")
        # gira si el maximo no esta en el primer punto explorado
        gira = thr_opt > suelo + 1e-9
        filas.append(dict(exp=NOMBRE.get(exp, exp), split=split, mec=MEC.get(mec, mec),
                          suelo=suelo, thr=thr_opt, f1=f1_opt, f1_025=f1_025,
                          delta=f1_opt - f1_025, exacto=gira,
                          bate_ref=f1_opt > REF[split]))
    return pd.DataFrame(filas)


if __name__ == "__main__":
    t = tabla()
    pd.set_option("display.width", 200)
    for split in ("test", "field_work"):
        s = t[t.split == split].sort_values("f1", ascending=False)
        print(f"\n{'='*96}\n{split}   (referencia {REF[split]})\n{'='*96}")
        print(f"{'bloque':26s} {'mecanismo':26s} {'suelo':>6s} {'thr':>5s} "
              f"{'F1 opt':>7s} {'F1@0,25':>8s} {'Δ':>6s}  exacto  bate ref")
        for _, r in s.iterrows():
            print(f"{r.exp:26s} {r.mec:26s} {r.suelo:6.2f} {r.thr:5.2f} "
                  f"{r.f1:7.2f} {r.f1_025:8.2f} {r.delta:+6.2f}   "
                  f"{'si' if r.exacto else 'COTA':4s}   {'SI' if r.bate_ref else ''}")
    t.to_csv(A / "optimos.csv", index=False)
    print("\n-> analisis/optimos.csv")
