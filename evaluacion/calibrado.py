#!/usr/bin/env python3
"""Reúne los barridos de confianza y devuelve las cifras calibradas de cada mecanismo.

Biblioteca que importan los guiones de tablas y figuras: carga los barridos de segmentación,
detección, expertos, arquitecturas de referencia y del Experimento 7, y para cada ejecución,
conjunto y mecanismo devuelve el mejor F1 con S-IoU al 50 % de solapamiento, el umbral en que
se alcanza y si ese máximo es solo una cota inferior, con el formato LaTeX ya preparado.
"""
from pathlib import Path
import pandas as pd

A = Path("/home/santana/Documents/docs_TFM/analisis")


def _carga():
    trozos = []
    for f in ("barrido_arriba.csv", "barrido_arriba_det.csv",
              "barrido_expertos.csv", "barrido_expertos_region.csv"):
        d = pd.read_csv(A / f)
        trozos.append(d[(d.metric == "S-IoU") & (d.overlap == 50)])
    d = pd.read_csv(A / "legacy_modelos.csv")
    d = d[(d.metric == "S-IoU") & (d.overlap == 50)].copy()
    d["experiment"] = "legacy:" + d["model"]
    d["mechanism"] = "dense_baseline"
    trozos.append(d[["experiment", "split", "mechanism", "score_thr", "f1",
                     "precision", "recall"]])
    # Mask2Former es el unico transformer que produce mascaras. Evaluarlo sobre
    # cajas lo penaliza mucho sobre campo, asi que se usa su barrido sobre
    # mascaras, reevaluado con la misma implementacion de S-IoU.
    m = pd.read_csv(A / "mask2former_mascaras.csv").copy()
    m["experiment"] = "legacy:mask2former_seg"
    m["mechanism"] = "dense_baseline"
    trozos.append(m[["experiment", "split", "mechanism", "score_thr", "f1",
                     "precision", "recall"]])
    d = pd.read_csv(A / "exp7_barrido.csv")
    d = d[(d.metric == "S-IoU") & (d.overlap == 50)].copy()
    d["mechanism"] = "dense_baseline"
    trozos.append(d[["experiment", "split", "mechanism", "score_thr", "f1",
                     "precision", "recall"]])
    return pd.concat(trozos, ignore_index=True)


_D = None


def optimo(exp, split, mec="dense_baseline"):
    """(f1, umbral, cota) o None. cota=True si el maximo cae en el suelo."""
    global _D
    if _D is None:
        _D = _carga()
    g = _D[(_D.experiment == exp) & (_D.split == split) & (_D.mechanism == mec)]
    if g.empty:
        return None
    g = g.sort_values("score_thr")
    i = g.f1.idxmax()
    return (float(g.loc[i, "f1"]), float(g.loc[i, "score_thr"]),
            bool(g.loc[i, "score_thr"] <= g.score_thr.min() + 1e-9))


def celda(exp, split, mec="dense_baseline", dec=2):
    """Cadena LaTeX con el valor y, si procede, la marca de cota."""
    r = optimo(exp, split, mec)
    if r is None:
        return "--"
    f1, thr, cota = r
    return f"{f1:.{dec}f}" + ("$^{\\dagger}$" if cota else "")


def umbral(exp, split, mec="dense_baseline"):
    r = optimo(exp, split, mec)
    if r is None:
        return "--"
    f1, thr, cota = r
    return ("$\\leq$" if cota else "") + f"{thr:.2f}"


if __name__ == "__main__":
    d = _carga()
    print(f"{len(d)} filas, {d.experiment.nunique()} ejecuciones")
    print(sorted(d.experiment.unique()))
