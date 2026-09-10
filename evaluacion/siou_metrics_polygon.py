"""Calcula las métricas IoU y S-IoU sobre geometría de polígono real mediante shapely.

Biblioteca que importan los demás scripts de evaluación: recibe cada polígono como una
lista plana de coordenadas, aplica el emparejamiento uno a uno del IoU y el emparejamiento
muchos a muchos del S-IoU, y devuelve TP, FP, FN, precisión, recall y F1 por imagen o
acumulados sobre el conjunto completo.
"""

from typing import List

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError as e:
    raise ImportError(
        "siou_metrics_polygon.py requiere shapely. Instala con: pip install shapely"
    ) from e

PolygonCoords = List[float]  # [x1,y1,x2,y2,...,xn,yn]


def _to_shapely(coords_flat: PolygonCoords) -> Polygon:
    """Convierte una lista plana de coordenadas a un Polygon de shapely,
    reparando auto-intersecciones menores con buffer(0) si hiciera falta
    (comun en poligonos generados a partir de mascaras/contornos)."""
    pts = list(zip(coords_flat[0::2], coords_flat[1::2]))
    if len(pts) < 3:
        return Polygon()  # poligono vacio/invalido -> area 0
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def poly_area(coords_flat: PolygonCoords) -> float:
    return _to_shapely(coords_flat).area


def iou(a_coords: PolygonCoords, b_coords: PolygonCoords) -> float:
    """IoU estandar (Ec. 1) entre dos poligonos."""
    a, b = _to_shapely(a_coords), _to_shapely(b_coords)
    if a.area == 0 or b.area == 0:
        return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _overlapping(target_coords: PolygonCoords, candidates_coords: List[PolygonCoords]):
    t = _to_shapely(target_coords)
    return [c for c in candidates_coords if t.intersects(_to_shapely(c))]


def evaluate_iou(pred_polys: List[PolygonCoords], gt_polys: List[PolygonCoords], iou_thr: float = 0.5) -> dict:
    """IoU estandar (Eq.1): matching greedy uno-a-uno, mejor-match unico.
    Las predicciones se asumen ya ordenadas por confianza descendente."""
    matched_gt = set()
    tp = 0
    for p in pred_polys:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_polys):
            if j in matched_gt:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j != -1:
            matched_gt.add(best_j)
            tp += 1

    fp = len(pred_polys) - tp
    fn = len(gt_polys) - len(matched_gt)
    return _prf1(tp, fp, fn)


def evaluate_soft_iou(pred_polys: List[PolygonCoords], gt_polys: List[PolygonCoords], iou_thr: float = 0.5) -> dict:
    """S-IoU (Ecs. 2 y 3 del paper): matching muchos-a-muchos via union.

    - TP/FP se determinan evaluando cada PREDICCION contra la union de
      TODOS los poligonos GT que solapan con ella (Ec. 2).
    - FN se determina evaluando cada poligono GT contra la union de TODAS
      las predicciones que solapan con el (Ec. 3).
    """
    tp, fp = 0, 0
    for p in pred_polys:
        s_lmatch = _overlapping(p, gt_polys)
        if not s_lmatch:
            fp += 1
            continue
        union_poly = unary_union([_to_shapely(g) for g in s_lmatch])
        p_poly = _to_shapely(p)
        union_area = union_poly.area
        inter_area = p_poly.intersection(union_poly).area
        s_iou_p = inter_area / union_area if union_area > 0 else 0.0
        if s_iou_p >= iou_thr:
            tp += 1
        else:
            fp += 1

    fn = 0
    for l in gt_polys:
        s_pmatch = _overlapping(l, pred_polys)
        if not s_pmatch:
            fn += 1
            continue
        union_poly = unary_union([_to_shapely(p) for p in s_pmatch])
        l_poly = _to_shapely(l)
        l_area = l_poly.area
        inter_area = l_poly.intersection(union_poly).area
        s_iou_l = inter_area / l_area if l_area > 0 else 0.0
        if s_iou_l < iou_thr:
            fn += 1

    return _prf1(tp, fp, fn)


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_detections(pred_polys: List[PolygonCoords], pred_scores: List[float],
                         gt_polys: List[PolygonCoords], iou_thr: float = 0.5,
                         score_thr: float = 0.0) -> dict:
    """Evalua un set de detecciones (de una imagen, o acumuladas de un
    dataset entero via evaluate_dataset) con IoU y S-IoU simultaneamente."""
    filtered = [(poly, s) for poly, s in zip(pred_polys, pred_scores) if s >= score_thr]
    filtered.sort(key=lambda x: -x[1])
    polys = [poly for poly, s in filtered]

    return {
        "iou": evaluate_iou(polys, gt_polys, iou_thr),
        "s_iou": evaluate_soft_iou(polys, gt_polys, iou_thr),
    }


def evaluate_dataset(per_image_predictions: List[tuple], per_image_gt: List[list],
                      iou_thr: float = 0.5, score_thr: float = 0.0) -> dict:
    """Acumula TP/FP/FN sobre TODAS las imagenes de un dataset (no
    promedia metricas por imagen, igual que siou_metrics.py)."""
    totals_iou = {"TP": 0, "FP": 0, "FN": 0}
    totals_siou = {"TP": 0, "FP": 0, "FN": 0}

    for (polys, scores), gt in zip(per_image_predictions, per_image_gt):
        r = evaluate_detections(polys, scores, gt, iou_thr=iou_thr, score_thr=score_thr)
        for k in ("TP", "FP", "FN"):
            totals_iou[k] += r["iou"][k]
            totals_siou[k] += r["s_iou"][k]

    return {
        "iou": _prf1(totals_iou["TP"], totals_iou["FP"], totals_iou["FN"]),
        "s_iou": _prf1(totals_siou["TP"], totals_siou["FP"], totals_siou["FN"]),
    }
