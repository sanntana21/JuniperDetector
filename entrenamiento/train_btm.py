"""Ejecuta el pipeline Branch-Train-Merge de expertos YOLO para la detección de Juniperus.

Parte del data.yaml del conjunto en formato YOLO y encadena, por etapas seleccionables desde la
línea de órdenes, el entrenamiento de un modelo semilla, la derivación de tres expertos con
regímenes de data augmentation distintos, las dos líneas base equiparadas en cómputo (modelo
único y ensemble no especializado) y la combinación de los expertos por WBF y por model soup
ponderado según su mAP50 en validación. Escribe esos pesos de fiabilidad en
merge_reliability_weights.json y la tabla comparativa sobre el conjunto de test en
final_comparison.json.
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuracion de los regimenes de augmentation por experto.
#
# Cada diccionario se pasa directamente como kwargs a YOLO.train(). Los
# parametros no listados aqui usan el default de Ultralytics.
#
# Diseñado para 3 expertos (dataset pequeño: 570 train imgs -> evitar
# fragmentar demasiado la divergencia). Los ejes elegidos, en base a la
# inspeccion visual del dataset (objetos pequeños, densos, fondo muy
# texturizado tipo roca/matorral):
#
#   E1_scale : robustez a escala / zoom (simula variacion de GSD)
#   E2_illum : robustez a iluminacion / contraste / sombra de ladera
#   E3_geom  : robustez a orientacion / oclusion parcial
# ---------------------------------------------------------------------------
EXPERT_CONFIGS = {
    "E1_scale": dict(
        scale=0.9, mosaic=0.3, hsv_v=0.2, degrees=0.0, shear=0.0, mixup=0.0,
    ),
    "E2_illum": dict(
        hsv_h=0.03, hsv_s=0.9, hsv_v=0.7, mosaic=0.1, degrees=0.0, shear=0.0,
    ),
    "E3_geom": dict(
        degrees=25.0, shear=10.0, perspective=0.0008, mosaic=0.2, hsv_v=0.2,
    ),
}

# Augmentation "estandar" usada en el seed, en el baseline denso y en el
# random ensemble de control (debe ser igual en los tres para que la
# comparacion sea justa).
STANDARD_AUG = dict(
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, mosaic=0.5, degrees=0.0, shear=0.0,
)


def get_yolo_class():
    """Import perezoso de ultralytics, con mensaje de error claro si falta."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Falta 'ultralytics'. Instala las dependencias con:\n"
            "    pip install ultralytics ensemble-boxes\n"
        ) from e
    return YOLO


def stage_seed(args):
    """Paso 0: entrena el modelo semilla en todo el train set."""
    YOLO = get_yolo_class()
    print(f"\n[SEED] entrenando {args.seed_epochs} epocas, base={args.base_weights}")

    model = YOLO(args.base_weights)
    model.train(
        data=args.data,
        epochs=args.seed_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.global_seed,
        project=args.project,
        name="seed",
        exist_ok=True,
        **STANDARD_AUG,
    )
    print(f"[SEED] listo -> {seed_weights_path(args)}")


def seed_weights_path(args):
    return str(Path(args.project) / "seed" / "weights" / "best.pt")


def stage_experts(args):
    """Paso 1-2-3: branch desde el seed y entrena cada experto con su propio
    regimen de augmentation, en paralelo conceptual (secuencial en este
    script, pero cada entrenamiento es independiente y podria lanzarse en
    procesos/GPUs distintos sin cambiar nada de la logica)."""
    YOLO = get_yolo_class()
    seed_w = seed_weights_path(args)
    if not Path(seed_w).exists():
        raise SystemExit(f"No se encuentra el seed entrenado en {seed_w}. Corre --stage seed primero.")

    for name, aug in EXPERT_CONFIGS.items():
        print(f"\n[EXPERT {name}] branch desde {seed_w}, augment={aug}")
        model = YOLO(seed_w)  # branch: clona los pesos del seed
        model.train(
            data=args.data,
            epochs=args.expert_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=args.global_seed,
            project=args.project,
            name=name,
            exist_ok=True,
            **aug,
        )
        print(f"[EXPERT {name}] listo -> {expert_weights_path(args, name)}")


def expert_weights_path(args, name):
    return str(Path(args.project) / name / "weights" / "best.pt")


def stage_dense(args):
    """Baseline denso compute-matched: mismo total de epocas que seed + un
    experto (porque en tiempo real los expertos corren en paralelo, no se
    suman), augment estandar, un solo modelo."""
    YOLO = get_yolo_class()
    total_epochs = args.seed_epochs + args.expert_epochs
    print(f"\n[DENSE] entrenando baseline denso, {total_epochs} epocas, augment estandar")

    model = YOLO(args.base_weights)
    model.train(
        data=args.data,
        epochs=total_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.global_seed,
        project=args.project,
        name="dense_baseline",
        exist_ok=True,
        **STANDARD_AUG,
    )
    print(f"[DENSE] listo -> {dense_weights_path(args)}")


def dense_weights_path(args):
    return str(Path(args.project) / "dense_baseline" / "weights" / "best.pt")


def stage_random_ensemble(args):
    """Baseline de control (replica el experimento §5.1 del paper BTM):
    N modelos branch-eados del MISMO seed, con el MISMO augmentation
    estandar, unica diferencia = semilla aleatoria. Si BTM-Forest no supera
    claramente a este control, la divergencia por augmentation no esta
    aportando nada mas alla de varianza de bagging generico."""
    YOLO = get_yolo_class()
    seed_w = seed_weights_path(args)
    if not Path(seed_w).exists():
        raise SystemExit(f"No se encuentra el seed entrenado en {seed_w}. Corre --stage seed primero.")

    n = len(EXPERT_CONFIGS)
    for i in range(n):
        name = f"random_{i}"
        run_seed = args.global_seed + 1000 + i
        print(f"\n[RANDOM {name}] branch desde {seed_w}, seed={run_seed}, augment=estandar")
        model = YOLO(seed_w)
        model.train(
            data=args.data,
            epochs=args.expert_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=run_seed,
            project=args.project,
            name=name,
            exist_ok=True,
            **STANDARD_AUG,
        )
        print(f"[RANDOM {name}] listo -> {random_weights_path(args, i)}")


def random_weights_path(args, i):
    return str(Path(args.project) / f"random_{i}" / "weights" / "best.pt")


# ---------------------------------------------------------------------------
# Merge: ensembling (WBF) y parameter averaging (model soup)
# ---------------------------------------------------------------------------

def get_reliability_weights(args, weight_paths, val_split="val"):
    """Calcula un peso por experto en base a su propio mAP50 en validacion.

    Esto sustituye al 'domain posterior' bayesiano del paper (que requiere
    saber de que dominio viene cada muestra). Aqui, sin dominios reales, se
    usa como proxy la fiabilidad global de cada experto en el val set: un
    experto que generaliza mejor pesa mas en el ensemble/promedio.
    """
    YOLO = get_yolo_class()
    weights = []
    for name, path in weight_paths.items():
        model = YOLO(path)
        metrics = model.val(data=args.data, split=val_split, verbose=False)
        map50 = float(metrics.box.map50)
        weights.append(map50)
        print(f"  {name}: mAP50(val) = {map50:.4f}")

    total = sum(weights)
    if total <= 0:
        # fallback: pesos uniformes si algo salio mal (todos mAP=0)
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def predict_ensemble_wbf(image_path, weight_paths, weights=None, iou_thr=0.55, skip_box_thr=0.1):
    """Ensembla las predicciones de varios modelos YOLO sobre una imagen via
    Weighted Box Fusion. Devuelve boxes normalizadas [0,1], scores, labels.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as e:
        raise SystemExit("Falta 'ensemble-boxes'. Instala con: pip install ensemble-boxes") from e

    YOLO = get_yolo_class()
    boxes_list, scores_list, labels_list = [], [], []

    for path in weight_paths.values():
        model = YOLO(path)
        r = model.predict(image_path, verbose=False)[0]
        b = r.boxes.xyxyn.cpu().numpy().tolist()  # normalizado, requerido por WBF
        s = r.boxes.conf.cpu().numpy().tolist()
        l = r.boxes.cls.cpu().numpy().tolist()
        boxes_list.append(b)
        scores_list.append(s)
        labels_list.append(l)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr,
    )
    return boxes, scores, labels


def average_weights(weight_paths, output_path, weights=None):
    """Model soup: promedia los state_dict de varios modelos YOLO que
    comparten arquitectura (todos branch-eados del mismo seed). El modelo
    resultante tiene el mismo costo de inferencia que un unico modelo.
    """
    try:
        import torch
    except ImportError as e:
        raise SystemExit("Falta 'torch'. Instalalo junto con ultralytics.") from e

    YOLO = get_yolo_class()

    paths = list(weight_paths.values())
    n = len(paths)
    if weights is None:
        weights = [1.0 / n] * n
    assert abs(sum(weights) - 1.0) < 1e-4, "Los pesos deben sumar 1."

    ckpts = [torch.load(p, map_location="cpu") for p in paths]
    state_dicts = [ckpt["model"].state_dict() for ckpt in ckpts]

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            raise SystemExit(
                f"El modelo #{i} ({paths[i]}) tiene una arquitectura de pesos distinta "
                f"al primero. El promedio de pesos solo es valido si todos los expertos "
                f"provienen del mismo seed/arquitectura."
            )

    avg_state = {}
    for key in ref_keys:
        stacked = [w * sd[key].float() for w, sd in zip(weights, state_dicts)]
        avg_state[key] = sum(stacked)

    # reconstruye un modelo YOLO con la arquitectura del primer checkpoint y
    # le carga los pesos promediados
    merged = YOLO(paths[0])
    merged.model.load_state_dict(avg_state, strict=False)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged.model}, output_path)
    print(f"[MERGE] pesos promediados guardados en {output_path}")
    return merged


def stage_merge(args):
    expert_paths = {name: expert_weights_path(args, name) for name in EXPERT_CONFIGS}
    missing = [p for p in expert_paths.values() if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Faltan pesos de expertos: {missing}. Corre --stage experts primero.")

    print("\n[MERGE] calculando pesos de fiabilidad por experto (proxy del domain posterior)")
    reliability_weights = get_reliability_weights(args, expert_paths, val_split="val")

    soup_path = str(Path(args.project) / "merged_soup" / "weights" / "best.pt")
    print("\n[MERGE] promediando pesos (model soup, weighted by reliability)")
    average_weights(expert_paths, soup_path, weights=reliability_weights)

    # guarda los pesos de fiabilidad para reutilizarlos en la evaluacion (WBF)
    weights_file = Path(args.project) / "merge_reliability_weights.json"
    weights_file.write_text(json.dumps(dict(zip(expert_paths.keys(), reliability_weights)), indent=2))
    print(f"[MERGE] pesos de fiabilidad guardados en {weights_file}")


# ---------------------------------------------------------------------------
# Evaluacion final sobre test (nunca tocado hasta este punto)
# ---------------------------------------------------------------------------

def evaluate_single(args, model_path, split="test"):
    YOLO = get_yolo_class()
    model = YOLO(model_path)
    metrics = model.val(data=args.data, split=split, verbose=False)
    return {"mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map)}


def compute_ap50(tp_list, conf_list, n_gt_total):
    """Calcula AP a IoU=0.5 (interpolacion continua, estilo VOC2010+/COCO)
    a partir de una lista de aciertos (True/False) y confianzas de todas
    las predicciones del dataset, ya emparejadas contra ground-truth por
    _iou_xyxy con umbral 0.5 (una gt solo puede emparejar una prediccion).

    Implementacion autocontenida a proposito: no depende de las funciones
    internas de ultralytics.utils.metrics, cuya firma cambia entre
    versiones de la libreria y romperia este script silenciosamente.
    """
    import numpy as np

    if n_gt_total == 0:
        return None  # no hay ground truth en el split: AP no esta definido

    if len(conf_list) == 0:
        return 0.0  # hay gt pero no hubo ninguna prediccion

    order = np.argsort(-np.array(conf_list))
    tp = np.array(tp_list, dtype=float)[order]
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # envolvente de precision (precision monotonamente no-creciente con el recall)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # integracion bajo la curva PR (recall va de 0 a max alcanzado)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0] if len(precision) else 1.0], precision))
    # np.trapz fue renombrado a np.trapezoid en numpy>=2.0; soporta ambas versiones
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    ap = trapz_fn(precision, recall)
    return float(ap)


def evaluate_ensemble_wbf(args, weight_paths, weights, split="test"):
    """Evalua el ensemble WBF sobre todas las imagenes del split indicado,
    calculando AP50 manualmente contra las anotaciones ground-truth en
    formato YOLO. Requiere 'ensemble-boxes' y las labels ya convertidas.

    Nota: esta implementacion calcula solo mAP50 (IoU=0.5), no el mAP50-95
    de COCO (que requiere repetir el matching a 10 umbrales de IoU). Se deja
    asi por simplicidad; mAP50 es suficiente para comparar las variantes de
    BTM entre si.
    """
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as e:
        raise SystemExit("Falta 'ensemble-boxes'. Instala con: pip install ensemble-boxes") from e

    YOLO = get_yolo_class()
    models = [YOLO(p) for p in weight_paths.values()]

    data_yaml_dir = Path(args.data).parent
    img_dir = data_yaml_dir / "images" / split
    lbl_dir = data_yaml_dir / "labels" / split
    img_paths = sorted(img_dir.glob("*"))

    all_tp, all_conf = [], []
    n_gt_total = 0

    for img_path in img_paths:
        boxes_list, scores_list, labels_list = [], [], []
        for m in models:
            r = m.predict(str(img_path), verbose=False)[0]
            boxes_list.append(r.boxes.xyxyn.cpu().numpy().tolist())
            scores_list.append(r.boxes.conf.cpu().numpy().tolist())
            labels_list.append(r.boxes.cls.cpu().numpy().tolist())

        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=0.55, skip_box_thr=0.1,
        )

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                x1, y1 = xc - bw / 2, yc - bh / 2
                x2, y2 = xc + bw / 2, yc + bh / 2
                gt_boxes.append([x1, y1, x2, y2])
        n_gt_total += len(gt_boxes)

        # ordena las predicciones fusionadas por confianza antes de emparejar,
        # asi una gt se asigna a la prediccion mas confiable que la solape
        order = sorted(range(len(fused_scores)), key=lambda i: -fused_scores[i])
        matched_gt = set()
        for i in order:
            box, score = fused_boxes[i], fused_scores[i]
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = _iou_xyxy(box, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= 0.5 and best_j != -1
            if is_tp:
                matched_gt.add(best_j)
            all_tp.append(is_tp)
            all_conf.append(score)

    ap50 = compute_ap50(all_tp, all_conf, n_gt_total)
    return {
        "mAP50": ap50,
        "mAP50-95": None,
        "note": "mAP50-95 no calculado (solo IoU@0.5); ver docstring de evaluate_ensemble_wbf",
    }


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def stage_evaluate(args):
    expert_paths = {name: expert_weights_path(args, name) for name in EXPERT_CONFIGS}
    random_paths = {f"random_{i}": random_weights_path(args, i) for i in range(len(EXPERT_CONFIGS))}
    dense_path = dense_weights_path(args)
    soup_path = str(Path(args.project) / "merged_soup" / "weights" / "best.pt")

    weights_file = Path(args.project) / "merge_reliability_weights.json"
    reliability_weights = None
    if weights_file.exists():
        reliability_weights = list(json.loads(weights_file.read_text()).values())

    results = {}

    if Path(dense_path).exists():
        print("\n[EVAL] denso baseline...")
        results["dense_baseline"] = evaluate_single(args, dense_path, split="test")

    if all(Path(p).exists() for p in expert_paths.values()):
        print("\n[EVAL] cada experto individual (referencia)...")
        for name, path in expert_paths.items():
            results[f"expert_{name}_solo"] = evaluate_single(args, path, split="test")

        print("\n[EVAL] BTM-Forest ensemble (WBF, pesos uniformes)...")
        results["btm_forest_ensemble_uniform"] = evaluate_ensemble_wbf(
            args, expert_paths, weights=None, split="test"
        )

        if reliability_weights:
            print("\n[EVAL] BTM-Forest ensemble (WBF, pesos por fiabilidad)...")
            results["btm_forest_ensemble_weighted"] = evaluate_ensemble_wbf(
                args, expert_paths, weights=reliability_weights, split="test"
            )

    if Path(soup_path).exists():
        print("\n[EVAL] BTM-Forest parameter average (model soup)...")
        results["btm_forest_soup"] = evaluate_single(args, soup_path, split="test")

    if all(Path(p).exists() for p in random_paths.values()):
        print("\n[EVAL] Random ensemble (control, WBF pesos uniformes)...")
        results["random_ensemble"] = evaluate_ensemble_wbf(
            args, random_paths, weights=None, split="test"
        )

    out_file = Path(args.project) / "final_comparison.json"
    out_file.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("COMPARACION FINAL (test set)")
    print("=" * 60)
    for name, m in results.items():
        map50 = m.get("mAP50")
        map5095 = m.get("mAP50-95")
        map50_str = f"{map50:.4f}" if map50 is not None else "N/A"
        map5095_str = f"{map5095:.4f}" if map5095 is not None else "N/A"
        print(f"  {name:<32} mAP50={map50_str}   mAP50-95={map5095_str}")
    print(f"\nResultados guardados en: {out_file}")


# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Pipeline Branch-Train-Merge para deteccion de Juniperus.")
    p.add_argument("--data", required=True, help="Ruta a data.yaml generado por prepare_dataset.py")
    p.add_argument("--stage", required=True,
                   choices=["seed", "experts", "dense", "random_ensemble", "merge", "evaluate", "all"])
    p.add_argument("--base-weights", default="yolov8n.pt",
                   help="Pesos preentrenados de partida (default: yolov8n.pt, COCO)")
    p.add_argument("--project", default="btm_juniperus", help="Carpeta de salida de todos los runs")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed-epochs", type=int, default=50)
    p.add_argument("--expert-epochs", type=int, default=50)
    p.add_argument("--global-seed", type=int, default=0)
    return p


def main():
    args = build_argparser().parse_args()

    stages_in_order = ["seed", "experts", "dense", "random_ensemble", "merge", "evaluate"]
    stage_fns = {
        "seed": stage_seed,
        "experts": stage_experts,
        "dense": stage_dense,
        "random_ensemble": stage_random_ensemble,
        "merge": stage_merge,
        "evaluate": stage_evaluate,
    }

    if args.stage == "all":
        for s in stages_in_order:
            stage_fns[s](args)
    else:
        stage_fns[args.stage](args)


if __name__ == "__main__":
    main()