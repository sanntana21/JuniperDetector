#!/usr/bin/env python3
"""Une la reevaluación general y la del Experimento 8 en un único fichero de métricas.

Concatena reeval_conf025.csv y reeval_exp8_conf025.csv, descarta los volcados de respaldo
que duplican al test de trabajo de campo y elimina las filas repetidas. Escribe
reeval_conf025_completo.csv, fuente de las tablas y figuras al umbral común 0,25.
"""
import pandas as pd
from pathlib import Path

A = Path("/home/santana/Documents/docs_TFM/analisis")

viejo = pd.read_csv(A / "reeval_conf025.csv")
nuevo = pd.read_csv(A / "reeval_exp8_conf025.csv")

# El backup a 0,25 del homogeneo duplica lo que ya esta en field_work
nuevo = nuevo[~nuevo.split.str.contains("backup")]
viejo = viejo[viejo.split != "test_conf025_backup"]

d = pd.concat([viejo, nuevo], ignore_index=True)
d = d.drop_duplicates(subset=["experiment", "split", "mechanism", "overlap",
                              "score_thr", "metric"])
d.to_csv(A / "reeval_conf025_completo.csv", index=False)

print(f"{len(viejo)} filas antiguas + {len(nuevo)} nuevas = {len(d)} tras deduplicar")
print(f"experimentos: {d.experiment.nunique()}")
for e in sorted(d.experiment.unique()):
    print("  ", e)
