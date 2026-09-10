#!/usr/bin/env python3
"""Barre el umbral de confianza hacia arriba en los expertos por tamaño evaluados en solitario.

Recorre los umbrales superiores al suelo de exportación de los expertos pequeño, mediano y
grande de las dos ejecuciones por tamaño, sobre el test fotointerpretado y el test de
trabajo de campo. Escribe barrido_expertos.csv con las filas de S-IoU al 50 % de
solapamiento, para que el calibrado lea cada experto en su propio umbral.
"""
import sys, os, json, glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/santana/Documents/TFM")
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import reevaluar                                            # noqa: E402

SRV = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup/experiments")
EXP8 = Path("/home/santana/Documents/docs_TFM/analisis/exp8/experiments")

RUNS = ["juniperus_seg_by_size_20260828_072323",
        "juniperus_seg_by_size_20260830_202803"]
MECS = ["expert_E_small_solo", "expert_E_medium_solo", "expert_E_large_solo"]

BAJOS = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22]
ALTOS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def suelo(path):
    d = json.load(open(path))
    mn = 1.0
    for im in d.get("images", []):
        sc = im.get("scores") or []
        if sc:
            mn = min(mn, min(sc))
    return mn


def localiza(run, split, mec):
    """Prefiere el volcado de exp8, que trae las reexportaciones."""
    for base in (EXP8, SRV):
        f = base / run / "predictions" / split / f"{mec}.json"
        if f.exists():
            return str(f)
    return None


def main():
    jobs = []
    for run in RUNS:
        for split in ("test", "field_work"):
            for mec in MECS:
                f = localiza(run, split, mec)
                if f is None:
                    continue
                s = suelo(f)
                thr = [t for t in BAJOS if t >= s] + ALTOS
                jobs.append((f, thr, (0.5,)))
    print(f"{len(jobs)} ficheros, {sum(len(j[1]) for j in jobs)} evaluaciones", flush=True)
    filas = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, res in enumerate(ex.map(reevaluar.eval_file, jobs), 1):
            filas.extend(res)
            print(f"  {i}/{len(jobs)}", flush=True)
    import pandas as pd
    d = pd.DataFrame(filas)
    d = d[d.metric == "S-IoU"]
    out = "/home/santana/Documents/docs_TFM/analisis/barrido_expertos.csv"
    d.to_csv(out, index=False)
    print("->", out, len(d), "filas")


if __name__ == "__main__":
    main()
