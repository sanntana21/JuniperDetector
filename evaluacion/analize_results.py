"""Evalúa las detecciones de caja con IoU y S-IoU a lo largo del umbral de confianza.

Lee las anotaciones COCO de referencia y el fichero process_results_bbox.json del test
fotointerpretado o del test de trabajo de campo, y acumula TP, FP y FN sobre todas las
imágenes. Guarda las curvas de F1 en results/ y aporta las funciones de S-IoU y de
emparejamiento que reutilizan los scripts de segmentación.
"""

import json
from shapely.ops import unary_union
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from utils import extract_number
from utils import xywh_to_xyxy
from utils import box_to_poly
import matplotlib.pyplot as plt
import argparse


def polygon_iou(poly1, poly2):
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def iou_single(boxA, boxB, eps=1e-9):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    
    #Computamos la interseccion de los bounding boxes
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    # Area del rectangulo que forman en su interseccion
    inter_area = inter_w * inter_h

    # Area de los BB por separado
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Area de la union = area los BB por separado
    # Menos el area total
    # |A ∪ B| = |A| + |B| − |A ∩ B|
    union = areaA + areaB - inter_area

    #Cuanto mayor interseccion y menor union mayor es el total
    return inter_area / (union + eps)


def iou_matrix(pred_boxes, gt_boxes):
    M = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            M[i, j] = iou_single(p, g)
    return M


def match_greedy_iou(iou_mat, iou_thr):
    M = iou_mat.copy()
    matched_pred = set()
    matched_gt = set()

    while M.size > 0:
        i, j = np.unravel_index(np.argmax(M), M.shape)
        if M[i, j] < iou_thr:
            break

        matched_pred.add(i)
        matched_gt.add(j)

        M[i, :] = -1
        M[:, j] = -1

    return matched_pred, matched_gt


# def match_greedy(iou_mat, iou_thr=0.5):
#     M = iou_mat.copy()
#     matched_pred = set()
#     matched_gt = set()

#     while True:
#         i, j = np.unravel_index(np.argmax(M), M.shape)
#         maxval = M[i, j]
#         if maxval < iou_thr:
#             break

#         matched_pred.add(i)
#         matched_gt.add(j)

#         M[i, :] = -1
#         M[:, j] = -1

#     return matched_pred, matched_gt


mode_map = {0: "Test", 1: "FieldWork"}

def siou_evaluation(pred_polys, gt_polys, siou_thr):
    """
    Calcula los True Positives (TP), False Positives (FP) y False Negatives (FN)
    utilizando la métrica S-IoU (Soft Intersection over Union), tal y como se
    define en el paper para la evaluación de especies polimórficas.

    A diferencia del IoU clásico, S-IoU NO realiza comparaciones uno a uno entre
    predicciones y anotaciones reales. En su lugar, evalúa la cobertura de área
    mediante un esquema asimétrico de dos fases:

    1) Evaluación de predicciones (TP / FP):
        Para cada predicción:
        - Si no se solapa con ninguna anotación real (GT), se considera un FP.
        - Si se solapa con una o más anotaciones:
            * Se calcula la unión de todas las anotaciones solapadas.
            * Se mide la fracción del área real (unión GT) cubierta por la predicción.
            * Si dicha fracción es mayor o igual que el umbral S-IoU, la predicción
                se considera un TP; en caso contrario, un FP.

        Esta fase corresponde a la Ecuación (2) del paper.

    2) Evaluación de anotaciones reales (FN):
        Para cada anotación real (GT):
        - Si no se solapa con ninguna predicción, se considera un FN.
        - Si se solapa con una o más predicciones:
            * Se calcula la unión de todas las predicciones solapadas.
            * Se mide la fracción del área de la anotación real cubierta por dicha unión.
            * Si esta fracción es inferior al umbral S-IoU, la anotación se considera FN.

        Esta fase corresponde a la Ecuación (3) del paper.

    La métrica S-IoU es asimétrica y permite relaciones muchos-a-muchos entre
    predicciones y anotaciones, lo que la hace adecuada para evaluar objetos
    polimórficos, colonias o anotaciones humanas inciertas.

    Args:
        pred_polys (List[shapely.geometry.Polygon]):
            Lista de polígonos correspondientes a las predicciones del modelo.

        gt_polys (List[shapely.geometry.Polygon]):
            Lista de polígonos correspondientes a las anotaciones reales (ground truth).

        siou_thr (float):
            Umbral S-IoU utilizado para decidir TP, FP y FN. Toma valores en [0, 1].

    Returns:
        Tuple[int, int, int]:
            - TP: número de True Positives.
            - FP: número de False Positives.
            - FN: número de False Negatives.
    """
    TP = 0
    FP = 0
    FN = 0

    for p in pred_polys:
        matched_gt = [g for g in gt_polys if p.intersects(g)]

        if not matched_gt:
            FP += 1
            continue

        gt_union = unary_union(matched_gt)
        siou_p = p.intersection(gt_union).area / gt_union.area

        if siou_p >= siou_thr:
            TP += 1
        else:
            FP += 1

    # ---------- Label evaluation (Eq. 3) ----------
    for g in gt_polys:
        matched_preds = [p for p in pred_polys if p.intersects(g)]

        if not matched_preds:
            FN += 1
            continue

        pred_union = unary_union(matched_preds)
        siou_l = g.intersection(pred_union).area / g.area

        if siou_l < siou_thr:
            FN += 1

    return TP, FP, FN



def run_analysis(
    mode : int = 0,
    confidence_threshold=0.5,
    metric_thr=0.5,
    metric="iou",   # "iou" or "siou"
    extension="",
    verbose=0
):
    extension = ""
    debug = False
    if mode == 1:
        field_work_data_dir = f"Field_Work_Data{extension}/External_Val_Data"
        results_path = f"{field_work_data_dir}/process_results_bbox.json"
        original_path = f"{field_work_data_dir}/Annotations/FieldWork_updated.json"
        images_path = f"{field_work_data_dir}/Images"
    else:
        photo_interpretation_test_dir = f"Photo_Interpretation_Data{extension}/Test"
        results_path = f"{photo_interpretation_test_dir}/process_results_bbox.json"
        original_path = f"{photo_interpretation_test_dir}/Annotations/Test_updated.json"
        images_path = f"{photo_interpretation_test_dir}/Images"

    with open(original_path) as f:
        coco_original = json.load(f)
        
    with open(results_path) as f:
        coco_results = json.load(f)
    
    import glob    
    lista_jpg = glob.glob(f"{images_path}/*.tif")
    lista_jpg = {i:j.split("/")[-1] for i,j in enumerate(sorted(lista_jpg,key=extract_number))}

    coco_results = [
        a for a in coco_results if a.get("score", 0) >= confidence_threshold
    ]
    anns = coco_original["annotations"]
    for a in anns:
        a['image_id'] = int(a['image_id'])
        a['file_name'] = lista_jpg[int(a['image_id'])]
        
    # if mode == 0:
    #     for a in coco_results:
    #         try:
    #             a['image_id'] = int(lista_jpg[int(a['image_id'])].split("_")[1:][0].split(".tif")[0:][0])
    #         except Exception:
    #             pass
    #     print("Nuevo id",a['image_id']) 
    if mode == 0:
        anns = [
            {
                **{k: v for k, v in ann.items() if k != "segmentation"},
                "id": extract_number(
                    coco_original["images"][int(ann["image_id"])]["file_name"]
                ),
                "image_id": extract_number(
                    coco_original["images"][int(ann["image_id"])]["file_name"]
                ),
                "file_name":coco_original["images"][int(ann["image_id"])]["file_name"]
            }
            for ann in coco_original["annotations"]
        ]
    
    image_ids = sorted(set(a["image_id"] for a in anns))

    TP_total = FP_total = FN_total = 0

    for img_id in image_ids:
        gt_anns = [a for a in anns if a["image_id"] == img_id]
        pred_anns = [a for a in coco_results if a["image_id"] == img_id]
        
        gt_boxes = [xywh_to_xyxy(a["bbox"]) for a in gt_anns]
        pred_boxes = [xywh_to_xyxy(a["bbox"]) for a in pred_anns]
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if debug:
            import os
            from PIL import Image  
            image_id = img_id
            
            img_path = os.path.join(images_path, f"Img_{image_id}.tif")

            # Load image
            image = Image.open(img_path)
            _, ax = plt.subplots(1, figsize=(10,10))
            ax.imshow(image)
            
            for ann in pred_anns:
                if "segmentation" in ann:
                    polygons = ann["segmentation"]
                    
                    for poly in polygons:
                        xs = poly[::2]
                        ys = poly[1::2]
                        ax.plot(
                            xs + [xs[0]], ys + [ys[0]],
                            linestyle="-", linewidth=1.5, color="red"
                        ) 
                else:
                    x, y, w, h = ann["bbox"]
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
                    )
                    ax.add_patch(rect)
                    
            for ann in gt_anns:
                if "segmentation" in ann:
                    for seg in ann["segmentation"]:
                        xs = seg[::2]
                        ys = seg[1::2]
                        ax.plot(
                            xs + [xs[0]], ys + [ys[0]],
                            linestyle="-", linewidth=1.5, color="blue"
                        ) 
                        
                else:
                    x, y, w, h = ann["bbox"]
                    rect = patches.Rectangle(
                        (x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none"
                    )
                    ax.add_patch(rect)
        
            ax.set_title(f"Image: {img_path} (id={image_id})")
            plt.axis("off")
            plt.show()

        if metric == "iou":
            # ---------- Standard IoU ----------
            if gt_boxes and pred_boxes:
                M = iou_matrix(pred_boxes, gt_boxes)
                matched_pred, matched_gt = match_greedy_iou(M, metric_thr)

                TP = len(matched_pred)
                FP = len(pred_boxes) - TP
                FN = len(gt_boxes) - TP
            else:
                TP = 0
                FP = len(pred_boxes)
                FN = len(gt_boxes)

        else:
            # ---------- S-IoU ----------
            gt_polys = [box_to_poly(b) for b in gt_boxes]
            pred_polys = [box_to_poly(b) for b in pred_boxes]

            TP, FP, FN = siou_evaluation(
                pred_polys,
                gt_polys,
                metric_thr
            )

        TP_total += TP
        FP_total += FP
        FN_total += FN

    precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) else 0
    recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) else 0
    F1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return F1, TP_total, FP_total, FN_total

def compute_precision_recall(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="Modo: 0 = Photo_Interpretation_Data, 1 = FieldWorkData",
    )
    args = parser.parse_args()
    mode = args.mode

    title = "PI" if mode == 0 else "FW"
    fixed_confidence = 0.9 if mode == 0 else 0.5

    confidence_values = [i / 10 for i in range(0, 11)]
    for metric_thr in [0.5,0.75]:

        results_iou = []
        results_siou = []

        # ---------- IoU ----------
        for confidence in confidence_values:
            F1, TP, FP, FN = run_analysis(
                mode=mode,
                confidence_threshold=confidence,
                metric_thr=metric_thr,
                metric="iou"
            )
            
            print("Iuo Iou")
            print(
                "f1",F1,"TP",TP,"FP",FP,"FN",FN
            )
            precision, recall = compute_precision_recall(TP, FP, FN)

            results_iou.append({
                "confidence": confidence,
                "F1": F1,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall
            })

        # ---------- SIoU ----------
        for confidence in confidence_values:
            F1, TP, FP, FN = run_analysis(
                mode=mode,
                confidence_threshold=confidence,
                metric_thr=metric_thr,
                metric="siou"
            )
            precision, recall = compute_precision_recall(TP, FP, FN)

            results_siou.append({
                "confidence": confidence,
                "F1": F1,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "precision": precision,
                "recall": recall
            })
        
        image = f"results/{title}_{metric_thr}"
        # ---------- Gráfica ----------
        plt.figure(figsize=(10, 6))
        plt.plot(confidence_values, [r["F1"] for r in results_iou],
                color='green',marker='o', label='IoU', linewidth=2)
        plt.plot(confidence_values, [r["F1"] for r in results_siou],
                color='gold',marker='s', label='SIoU', linewidth=2)
        plt.xlabel("Confidence threshold")
        plt.ylabel("F1 score")
        plt.title(f"{title} Evaluation Metric Threshold {metric_thr}")
        plt.legend()
        plt.grid(True)
        plt.savefig(image + "_zoom.png", dpi=300, bbox_inches="tight")
        
        confidence_values_pct = [c * 100 for c in confidence_values]

        plt.plot(confidence_values_pct, [r["F1"] * 100 for r in results_iou], marker='o', color='green',linewidth=2,label='IoU')
        plt.plot(confidence_values_pct, [r["F1"] * 100 for r in results_siou], marker='s', color='gold',linewidth=2,label='SIoU')

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.xticks([0, 20, 40, 60, 80, 100])
        plt.yticks([0, 20, 40, 60, 80, 100])

        plt.xlabel("Confidence threshold (%)")
        plt.ylabel("F1 score (%)")
        plt.savefig(image + "_paper.png", dpi=300, bbox_inches="tight")
        

        # ---------- Selección de resultados ----------
        best_iou = max(results_iou, key=lambda x: x["F1"])
        best_siou = max(results_siou, key=lambda x: x["F1"])

        fixed_iou = next(r for r in results_iou if r["confidence"] == fixed_confidence)
        fixed_siou = next(r for r in results_siou if r["confidence"] == fixed_confidence)

        # ---------- Tabla resumen ----------
        print("\n" + "=" * 90)
        print(f"RESUMEN DE RESULTADOS ({title} Confidence {metric_thr})")
        print("=" * 90)
        print(f"{'Metric':<8} | {'Conf':<5} | {'TP':<5} | {'FP':<5} | {'FN':<5} | "
            f"{'Precision':<9} | {'Recall':<6} | {'F1':<6}")
        print("-" * 90)

        def print_row(metric, r):
            print(f"{metric:<8} | {r['confidence']:<5.2f} | {r['TP']:<5} | {r['FP']:<5} | {r['FN']:<5} | "
                f"{r['precision']*100:<9.3f} | {r['recall']*100:<6.3f} | {r['F1']*100:<6.3f}")

        print_row("IoU*", fixed_iou)
        print_row("SIoU*", fixed_siou)
        print("-" * 90)
        print_row("IoU", best_iou)
        print_row("SIoU", best_siou)
        print("=" * 90)
        print("* Confianza fija según modo")