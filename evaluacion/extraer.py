#!/usr/bin/env python3
"""Consolida las métricas y la configuración de todos los experimentos del servidor.

Recorre los directorios de ejecución, extrae la configuración de la cabecera del registro
de entrenamiento y lee los ficheros final_comparison de ambos conjuntos de evaluación.
Escribe metricas.csv con IoU, S-IoU y mAP por mecanismo y umbral, participacion.csv con los
indicadores de participación de los expertos y configuraciones.csv con los parámetros.
"""
import json, re, os, sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup/experiments")
OUT  = Path("/home/santana/Documents/docs_TFM/analisis")
OUT.mkdir(exist_ok=True)

def parse_log(p):
    """Extrae configuracion de la cabecera del log de entrenamiento."""
    cfg = {}
    try:
        txt = p.read_text(errors="replace")
    except Exception:
        return cfg
    m = re.search(r"\[VARIANT\] usando '([^']+)': (.+)", txt)
    if m:
        cfg["variant"] = m.group(1)
        cfg["archs"] = m.group(2).strip()
    m = re.search(r"\[SEED:([^\]]+)\] entrenando (\d+) epocas", txt)
    if m:
        cfg["seed_arch"], cfg["seed_epochs"] = m.group(1), int(m.group(2))
    m = re.search(r"\[EXPERT:[^\]]+\] entrenando (\d+) epocas", txt)
    if m:
        cfg["expert_epochs"] = int(m.group(1))
    # primera linea de 'engine/trainer:' con la config completa de ultralytics
    m = re.search(r"engine/trainer: (.+)", txt)
    if m:
        line = m.group(1)
        for k in ("epochs", "imgsz", "batch", "patience", "data", "model", "task"):
            mm = re.search(rf"\b{k}=([^,]+)", line)
            if mm:
                cfg[f"ul_{k}"] = mm.group(1).strip()
    # umbral de confianza de prediccion, si se registro
    for pat in (r"--predict-conf[= ]([\d.]+)", r"predict_conf[=:] ?([\d.]+)",
                r"\[PREDICT\][^\n]*conf[= ]([\d.]+)"):
        mm = re.search(pat, txt)
        if mm:
            cfg["predict_conf"] = float(mm.group(1)); break
    cfg["log_bytes"] = p.stat().st_size
    return cfg

rows, part_rows, cfg_rows = [], [], []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue
    logs = list(d.glob("*.log"))
    cfg = parse_log(logs[0]) if logs else {}
    cfg["experiment"] = d.name
    # epocas realmente ejecutadas
    ts = d / "training_summary.json"
    if ts.exists():
        try:
            summ = json.loads(ts.read_text())
            for run, info in summ.items():
                if isinstance(info, dict) and "epochs" in info:
                    cfg[f"epochs_{run}"] = info["epochs"]
        except Exception as e:
            print("  ! training_summary ilegible en", d.name, e, file=sys.stderr)
    cfg_rows.append(cfg)

    for split in ("test", "field_work"):
        f = d / f"final_comparison_{split}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        for mech, blob in data.items():
            if mech == "participation_insights":
                part_rows.append({"experiment": d.name, "split": split,
                                  **{k: json.dumps(v) if isinstance(v, (dict, list)) else v
                                     for k, v in blob.items()}})
                continue
            if not isinstance(blob, dict):
                continue
            prim = blob.get("primary_iou_siou", {})
            for thr, mblob in prim.items():
                for metric in ("iou", "s_iou"):
                    v = mblob.get(metric)
                    if not v:
                        continue
                    rows.append({
                        "experiment": d.name, "split": split, "mechanism": mech,
                        "threshold": thr, "metric": "S-IoU" if metric == "s_iou" else "IoU",
                        "TP": v.get("TP"), "FP": v.get("FP"), "FN": v.get("FN"),
                        "precision": 100 * v.get("precision", float("nan")),
                        "recall":    100 * v.get("recall", float("nan")),
                        "f1":        100 * v.get("f1", float("nan")),
                        "mAP50":     blob.get("secondary_map", {}).get("mAP50"),
                        "mAP50_95":  blob.get("secondary_map", {}).get("mAP50-95"),
                    })

pd.DataFrame(rows).to_csv(OUT / "metricas.csv", index=False)
pd.DataFrame(part_rows).to_csv(OUT / "participacion.csv", index=False)
pd.DataFrame(cfg_rows).to_csv(OUT / "configuraciones.csv", index=False)
print(f"metricas: {len(rows)} filas | participacion: {len(part_rows)} | configs: {len(cfg_rows)}")
