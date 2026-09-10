#!/usr/bin/env python3
"""Genera los fragmentos LaTeX de todas las tablas de la memoria a partir de los CSV de análisis.

Lee los barridos de confianza, las reevaluaciones y los inventarios de ejecuciones,
aplica sobre ellos el criterio calibrado de calibrado.py y escribe un fichero .tex por
tabla en latex/tablas, listo para incluirse en el documento. Admite como argumentos
los nombres de las tablas que se quieren regenerar; sin argumentos las genera todas.
"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import pandas as pd

A = Path("/home/santana/Documents/docs_TFM/analisis")
T = Path("/home/santana/Documents/docs_TFM/latex/tablas")
T.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- utilidades
SIGLAS = ""   # PI y FW se declaran una sola vez, en el marco de lectura del cap. 7


def dec(x):
    """Numero con dos cifras decimales."""
    return f"{x:.2f}"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


_NUM = re.compile(r"(?<![A-Za-z0-9])(\d+)\.(\d+)(?![A-Za-z0-9])")


def comas(texto):
    """Coma decimal en todas las cifras, sin tocar 4.4cm ni 2.5pt (§7)."""
    return _NUM.sub(lambda m: m.group(1) + "{,}" + m.group(2), texto)


def escribe(nombre, contenido):
    (T / f"{nombre}.tex").write_text(comas(contenido).rstrip() + "\n", encoding="utf-8")
    print("  ->", nombre + ".tex")


def tabla(cabeceras, filas, colspec, caption, label, small=True,
          tabcolsep=None, notas=None, midrules=()):
    """Construye un entorno table+tabular con booktabs."""
    out = ["\\begin{table}[H]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    out.append({True: "\\small", False: "\\footnotesize",
                "script": "\\scriptsize"}[small])
    if tabcolsep:
        out.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\toprule")
    out.append(" & ".join(cabeceras) + " \\\\")
    out.append("\\midrule")
    for i, f in enumerate(filas):
        if i in midrules:
            out.append("\\midrule")
        out.append(" & ".join(str(x) for x in f) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if notas:
        out.append(f"\\\\[2pt]\\footnotesize {notas}")
    out.append("\\end{table}")
    return "\n".join(out)


NOMBRE_MEC = {
    "dense_baseline":              "Modelo único",
    "btm_forest_ensemble_uniform": "\\textit{Ensemble} de expertos",
    "btm_forest_ensemble_weighted":"\\textit{Ensemble} ponderado",
    "btm_forest_soup":             "\\textit{Model merging}",
    "random_ensemble":             "\\textit{Ensemble} no especializado",
    "expert_E_small_solo":         "Experto pequeño (solo)",
    "expert_E_medium_solo":        "Experto mediano (solo)",
    "expert_E_large_solo":         "Experto grande (solo)",
    "expert_E1_scale_solo":        "Experto de escala (solo)",
    "expert_E2_illum_solo":        "Experto de iluminación (solo)",
    "expert_E3_geom_solo":         "Experto de geometría (solo)",
    "expert_E_region0_solo":       "Experto región 0 (solo)",
    "expert_E_region1_solo":       "Experto región 1 (solo)",
}

PAPER = {   # trabajo de referencia: (P, R, F1) por split/metrica/umbral
    ("test", "IoU", 50):   (87.93, 81.96, 84.84),
    ("test", "IoU", 75):   (76.47, 71.59, 73.95),
    ("test", "S-IoU", 50): (88.55, 87.20, 87.87),
    ("test", "S-IoU", 75): (85.29, 84.25, 84.77),
    ("field_work", "IoU", 50):   (74.32, 70.56, 72.39),
    ("field_work", "IoU", 75):   (48.82, 47.03, 47.91),
    ("field_work", "S-IoU", 50): (76.61, 77.11, 76.86),
    ("field_work", "S-IoU", 75): (66.88, 69.83, 68.32),
}
PAPER_TPFPFN = {
    ("test", "IoU", 50):   (568, 78, 125),
    ("test", "IoU", 75):   (494, 152, 196),
    ("test", "S-IoU", 50): (572, 74, 84),
    ("test", "S-IoU", 75): (551, 95, 103),
    ("field_work", "IoU", 50):   (1268, 438, 529),
    ("field_work", "IoU", 75):   (833, 873, 938),
    ("field_work", "S-IoU", 50): (1307, 399, 388),
    ("field_work", "S-IoU", 75): (1141, 565, 493),
}


# =============================================================== inventario
def t_inventario():
    filas = [
        ["Modelo único", "--", "Ambas",
         "DETR, Mask2Former", "10"],
        ["Divergencia inducida", "Sintético: aumento", "Detección",
         "YOLO n (3 gen.)", "2"],
        ["Espec. por tamaño", "Real: área del arbusto", "Segm.",
         "YOLO-seg n/l", "4 (+2)"],
        ["Espec. geográfica", "Real: longitud", "Segm.",
         "YOLO-seg n", "2"],
        ["\\textit{Slicing}", "--", "Segm.",
         "YOLO-seg n", "3"],
        ["Umbral fijo", "--", "Segm.",
         "Todos", "--"],
        ["Más señal", "Datos: copy-paste, destilación", "Segm.",
         "YOLO-seg n", "2"],
    ]
    return tabla(
        ["Bloque", "Eje o palanca", "Tarea", "Modelos", "N"],
        filas,
        "@{}lllll@{}",
        "Los siete bloques experimentales. La columna «eje o palanca» indica el "
        "criterio con el que se induce la divergencia entre expertos, o la "
        "variable sobre la que actúa el bloque cuando no hay expertos.",
        "tab:inventario", small="script",
        tabcolsep="3pt",
        notas="Los entrenamientos entre paréntesis de la especialización por tamaño "
              "corresponden a la condición de entrenamiento conjunto analizada en "
              "la Sección~\\ref{sec:condiciones}.")


# ================================================= exploracion arquitecturas
def t_legacy():
    d = pd.read_csv(A / "curvas_legacy.csv")
    v = d[d["mode"] == "val"]
    # (experimento, log, arquitectura, cabezal, extractor, configuracion)
    SEL = [
        ("exp_v1", "20251027_093440.log.json", "DETR", "--", "ResNet-50", "150 épocas"),
        ("exp_v1", "20251107_080715.log.json", "Deformable DETR", "--", "ResNet-50", "150 épocas"),
        ("exp_tiff_old", "20260120_150049.log.json", "Deformable DETR", "--", "ResNet-50", "300 épocas"),
        ("exp_tiff_old", "20260121_160401.log.json", "Deformable DETR", "--", "ResNet-50", "600 épocas"),
        ("exp_tiff", "20260122_145259.log.json", "Deformable DETR", "--", "ResNet-50", "1200 épocas"),
        ("exp_tiff_data_augmentation_cicled_lr", "20260206_115955.log.json",
         "Deformable DETR", "--", "ResNet-50", "1200 ép. + \\textit{data augment.}"),
        ("experimento_mask2former", "20260210_112216.log.json",
         "Mask2Former", "--", "Swin-T", "800 épocas"),
        ("exp_co_deformable_detr_swimL_juniper", "20260328_131550.log.json",
         "Co-DETR", "Co-Deform.", "Swin-L", "100 ép., 300 cons."),
        ("exp_co_dino_swimL_juniper_v1", "20260303_084940.log.json",
         "Co-DETR", "Co-DINO", "Swin-L", "80 ép., 900 cons."),
        ("exp_co_dino_swimL_juniper", "20260309_181154.log.json",
         "Co-DETR", "Co-DINO", "Swin-L", "100 ép., 900 cons."),
    ]
    filas = []
    for exp, fich, arq, cab, backbone, nota in SEL:
        s_ = v[(v.experimento == exp) & (v.fichero == fich)]
        if s_.empty:
            continue
        i = s_["bbox_mAP_50"].idxmax()
        segm = s_.loc[i, "segm_mAP_50"]
        filas.append([arq, cab, backbone, nota, int(s_.loc[i, "epoch"]),
                      f"{100*s_.loc[i,'bbox_mAP_50']:.1f}",
                      f"{100*s_.loc[i,'bbox_mAP']:.1f}",
                      f"{100*segm:.1f}" if pd.notna(segm) else "--"])
    return tabla(
        ["Arquitectura", "Cabezal", "\\textit{Backbone}", "Config.", "Ép.",
         "mAP@50", "mAP@50-95", "másc."],
        filas,
        "@{}llllrrrr@{}",
        "Exploración de arquitecturas. Para cada configuración se indica la época "
        "en que alcanzó su mejor mAP@50 sobre el conjunto de validación y las "
        "métricas en ese punto.",
        "tab:legacy", small="script",
        tabcolsep="1.5pt",
        notas="«másc.» es el mAP@50 de máscara, solo definido para los modelos que "
              "producen segmentación. Las filas de 300 y de 600 épocas de Deformable "
              "DETR son reanudaciones sucesivas de un mismo entrenamiento.")


# ============================================ helpers para tablas de metricas
def carga_reeval():
    d = pd.read_csv(A / "reeval_conf025_completo.csv")
    d["mecanismo"] = d["mechanism"].map(NOMBRE_MEC).fillna(d["mechanism"])
    return d


def bloque_prf(d, exp, split, mecanismos, overlap=50, metrica="S-IoU"):
    """Devuelve filas [nombre, P, R, F1] ordenadas segun `mecanismos`."""
    sub = d[(d.experiment == exp) & (d.split == split) &
            (d.overlap == overlap) & (d.metric == metrica)]
    filas = []
    for m in mecanismos:
        r = sub[sub.mechanism == m]
        if r.empty:
            continue
        r = r.iloc[0]
        filas.append([NOMBRE_MEC.get(m, m), f"{r.precision:.1f}", f"{r.recall:.1f}",
                      f"{r.f1:.2f}"])
    return filas


def t_experimento(exp, split, mecanismos, caption, label, con_referencia=True):
    """Tabla IoU + S-IoU (@50) de un experimento y un split."""
    d = carga_reeval()
    sub = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50)]
    filas = []
    if con_referencia:
        pi = PAPER[(split, "IoU", 50)]
        ps = PAPER[(split, "S-IoU", 50)]
        filas.append([r"\textit{Referencia (Mask R-CNN)}",
                      f"\\textit{{{pi[2]:.2f}}}", f"\\textit{{{ps[0]:.1f}}}",
                      f"\\textit{{{ps[1]:.1f}}}", f"\\textit{{{ps[2]:.2f}}}"])
    mejor = -1
    cuerpo = []
    for m in mecanismos:
        ri = sub[(sub.mechanism == m) & (sub.metric == "IoU")]
        rs = sub[(sub.mechanism == m) & (sub.metric == "S-IoU")]
        if ri.empty or rs.empty:
            continue
        ri, rs = ri.iloc[0], rs.iloc[0]
        cuerpo.append([NOMBRE_MEC.get(m, m), f"{ri.f1:.2f}", f"{rs.precision:.1f}",
                       f"{rs.recall:.1f}", rs.f1])
        mejor = max(mejor, rs.f1)
    for f in cuerpo:
        f[4] = (f"\\textbf{{{f[4]:.2f}}}" if abs(f[4] - mejor) < 1e-9 else f"{f[4]:.2f}")
    filas += cuerpo
    return tabla(
        ["Mecanismo", "IoU F1", "S-IoU P", "S-IoU R", "S-IoU F1"],
        filas, "@{}lrrrr@{}", caption, label, tabcolsep="5pt",
        midrules=(1,) if con_referencia else ())


def t_experimento_cal(exp, split, mecanismos, caption, label):
    """Desglose de un experimento con cada mecanismo en su umbral optimo."""
    d = _cal._carga()
    ref = PAPER[(split, "S-IoU", 50)]
    filas = [[r"\textit{Referencia (Mask R-CNN)}", "--",
              f"\\textit{{{ref[0]:.1f}}}", f"\\textit{{{ref[1]:.1f}}}",
              f"\\textit{{{ref[2]:.2f}}}"]]
    cuerpo, mejor = [], -1
    for m in mecanismos:
        r = _cal.optimo(exp, split, m)
        if r is None:
            continue
        g = d[(d.experiment == exp) & (d.split == split) & (d.mechanism == m)]
        fila = g.loc[g.f1.idxmax()]
        cuerpo.append([NOMBRE_MEC.get(m, m), f"{r[1]:.2f}",
                       f"{fila.precision:.1f}", f"{fila.recall:.1f}", r[0]])
        mejor = max(mejor, r[0])
    for f in cuerpo:
        f[4] = (f"\\textbf{{{f[4]:.2f}}}" if abs(f[4] - mejor) < 1e-9 else f"{f[4]:.2f}")
    filas += cuerpo
    return tabla(["Mecanismo", "Umbral", "P", "R", "F1"], filas,
                 "@{}lrrrr@{}", caption, label, tabcolsep="5pt", midrules=(1,))


# ============================================== comparativa con la referencia
def t_comparativa(exp, mecanismos, caption, label):
    """Replica el formato de la tabla de resultados del trabajo de referencia:
    conjunto x metrica x umbral, con P/R/F1 de cada mecanismo."""
    d = carga_reeval()
    cols = ["Conjunto", "Métrica", "Umbral"]
    for m in mecanismos:
        cols.append("\\multicolumn{3}{c}{" + NOMBRE_MEC.get(m, m).split(" (")[0] + "}")
    ncols = 3 + 3 * (len(mecanismos) + 1)
    lin = ["@{}llc", "@{\\hspace{5pt}}ccc" * (len(mecanismos) + 1), "@{}"]
    out = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{2pt}",
           "\\begin{tabular}{" + "".join(lin) + "}", "\\toprule"]
    cab = ["\\multirow{2}{*}{Conjunto}", "\\multirow{2}{*}{Métrica}", "\\multirow{2}{*}{Umbral}",
           "\\multicolumn{3}{c}{Referencia}"]
    for m in mecanismos:
        cab.append("\\multicolumn{3}{c}{" + NOMBRE_MEC.get(m, m).split(" (")[0] + "}")
    out.append(" &\n".join(cab) + " \\\\")
    reglas = []
    for i in range(len(mecanismos) + 1):
        a = 4 + 3 * i
        reglas.append(f"\\cmidrule(lr){{{a}-{a+2}}}")
    out.append("".join(reglas))
    out.append(" & & & " + " & ".join(["P", "R", "F1"] * (len(mecanismos) + 1)) + " \\\\")
    out.append("\\midrule")
    for si, (split, etiq) in enumerate([("test", "Test\\\\(PI)"), ("field_work", "Campo\\\\(FW)")]):
        if si:
            out.append("\\midrule")
        primero = True
        for met in ("IoU", "S-IoU"):
            for thr in (50, 75):
                celda = ("\\multirow{4}{*}{\\makecell[l]{" + etiq + "}}") if primero else ""
                primero = False
                p = PAPER[(split, met, thr)]
                fila = [celda, met, f"{thr}\\,\\%",
                        f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}"]
                for m in mecanismos:
                    r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == thr) &
                          (d.metric == met) & (d.mechanism == m)]
                    if r.empty:
                        fila += ["--", "--", "--"]
                    else:
                        r = r.iloc[0]
                        fila += [f"{r.precision:.1f}", f"{r.recall:.1f}", f"{r.f1:.1f}"]
                out.append(" & ".join(fila) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


# ============================================== capa calibrada (barrido propio)
import calibrado as _cal                                     # noqa: E402

CANONICAS = [
    ("legacy:mask2former_seg",                      "Arquitectura \\textit{transformer}", True),
    ("juniperus_seg_homogeneous_20260905_105741",   "\\textit{Data augmentation}, homogénea",     False),
    ("juniperus_seg_heterogeneous_20260905_111758", "\\textit{Data augmentation}, heterogénea",   False),
    ("juniperus_seg_by_size_20260830_202803",       "Tamaño del arbusto",                         False),
    ("juniperus_seg_by_region_20260831_073029",     "Región geográfica",                          False),
    ("copypaste_test",                              "\\textit{Copy-paste}",                       True),
    ("distilled",                                   "Destilación",                                True),
]
MEC_GLOBAL = [("dense_baseline", "Único"),
              ("btm_forest_ensemble_uniform", "Ens."),
              ("btm_forest_ensemble_weighted", "Pond."),
              ("btm_forest_soup", "Fus."),
              ("random_ensemble", "\\makecell{No\\\\espec.}")]


def _cal_celda(exp, split, mec, dec=1):
    r = _cal.optimo(exp, split, mec)
    if r is None:
        return "--"
    f1, thr, cota = r
    return f"{f1:.{dec}f}"


def t_resumen_global_cal():
    """Comparacion global con cada mecanismo en su propio umbral optimo."""
    filas = []
    for exp, nom, unico in CANONICAS:
        fila = [nom]
        for split in ("test", "field_work"):
            if unico:
                v = _cal_celda(exp, split, "dense_baseline")
                fila.append(f"\\multicolumn{{5}}{{c}}{{{v}}}")
                continue
            for mec, _ in MEC_GLOBAL:
                fila.append(_cal_celda(exp, split, mec))
        filas.append(fila)
    cab = " & ".join(e for _, e in MEC_GLOBAL)
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Comparación global. Una configuración por experimento, la "
           "canónica de cada bloque.}",
           "\\label{tab:resumen_global}", "\\scriptsize",
           "\\setlength{\\tabcolsep}{2pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.5cm}"
           "@{\\hspace{4pt}}rrrrr@{\\hspace{4pt}}rrrrr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Experimento} & "
           "\\multicolumn{5}{c}{Test (PI)} & \\multicolumn{5}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
           " & " + cab + " & " + cab + " \\\\",
           "\\midrule"]
    for f in filas:
        out.append(" & ".join(f) + " \\\\")
    out += ["\\midrule",
            "\\textit{Referencia} & "
            "\\multicolumn{5}{c}{\\textit{87.87}} & \\multicolumn{5}{c}{\\textit{76.86}} \\\\",
            "\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize Tres bloques no construyen expertos y son un "
            "solo modelo, cuya cifra ocupa las cinco columnas. El de \\textit{transformers} es "
            "Mask2Former Swin-T sobre sus máscaras, la única configuración de esa familia que "
            "las produce.",
            "\\end{table}"]
    return "\n".join(out)


def t_referencias_cal():
    """Regla de seleccion sobre cifras calibradas: la mejor sobre FW."""
    UNI, ENS = "dense_baseline", "btm_forest_ensemble_uniform"
    A_UNI, A_ENS = "Único", "\\textit{Ens.}"
    BLOQUES = [
        ("Arquitectura \\textit{transformer}", A_UNI, UNI, [
            ("legacy:codetr_r50",        "Deformable DETR R50"),
            ("legacy:codetr_r50_aug",    "Def. DETR R50 + \\textit{data augment.}"),
            ("legacy:codetr_r50_v1",     "Def. DETR R50, inicial"),
            ("legacy:co_deform_swinL",   "Co-Deformable-DETR Swin-L"),
            ("legacy:co_dino_swinL",     "Co-DINO Swin-L"),
            ("legacy:co_dino_swinL_v1",  "Co-DINO Swin-L, v1"),
            ("legacy:mask2former_swinT", "Mask2Former Swin-T (cajas)"),
            ("legacy:mask2former_seg",   "Mask2Former Swin-T (máscaras)")]),
        ("\\textit{Data augment.}, homogénea", A_ENS, ENS, [
            ("juniperus_btm_homogeneous_20260824_103026", "Detección"),
            ("juniperus_seg_homogeneous_20260905_105741", "Segmentación")]),
        ("\\textit{Data augment.}, heterogénea", A_ENS, ENS, [
            ("juniperus_btm_heterogeneous_20260824_091248", "Detección"),
            ("juniperus_seg_heterogeneous_20260905_111758", "Segmentación")]),
        ("Tamaño del arbusto", A_ENS, ENS, [
            ("juniperus_seg_by_size_20260830_163740", "División natural, 50 ép."),
            ("juniperus_seg_by_size_20260830_164608", "Cantidad igualada, 50 ép."),
            ("juniperus_seg_by_size_20260830_202803", "División natural, 100 ép."),
            ("juniperus_seg_by_size_20260830_221329", "Modelo grande, 100 ép.")]),
        ("Región geográfica", A_ENS, ENS, [
            ("juniperus_seg_by_region_20260828_114011", "Primera ejecución"),
            ("juniperus_seg_by_region_20260831_073029", "Ejecución canónica")]),
        ("\\textit{Copy-paste}", A_UNI, UNI, [
            ("copypaste_test", "Modelo único aumentado")]),
        ("Destilación", A_UNI, UNI, [
            ("distilled", "Estudiante del \\textit{ensemble}")]),
    ]

    def mejor(variantes, split, mec):
        cand = []
        for e, n in variantes:
            r = _cal.optimo(e, split, mec)
            if r is not None:
                cand.append((n, r))
        if not cand:
            return "--"
        n, (f1, thr, cota) = max(cand, key=lambda c: c[1][0])
        return f"{n}, {f1:.2f}"

    filas = []
    for bloque, arb, mec, variantes in BLOQUES:
        filas.append([bloque, arb, mejor(variantes, "test", mec),
                      "\\textbf{" + mejor(variantes, "field_work", mec) + "}"])
    return tabla(
        ["Bloque", "Árbitro", "Mejor sobre PI", "Mejor sobre FW, que es la canónica"],
        filas,
        "@{}>{\\raggedright\\arraybackslash}p{2.7cm}"
        ">{\\raggedright\\arraybackslash}p{1.2cm}"
        ">{\\raggedright\\arraybackslash}p{3.8cm}"
        ">{\\raggedright\\arraybackslash}p{4.0cm}@{}",
        "Configuración canónica de cada bloque. Para cada uno se da la que mejor "
        "rinde sobre el test fotointerpretado y la que mejor rinde sobre el de "
        "trabajo de campo, medidas sobre el mecanismo que contrasta la hipótesis "
        "del bloque.",
        "tab:referencias", small="script", tabcolsep="3pt",
        notas="Los dos bloques de ampliación de la señal tienen una sola "
              "configuración cada uno, de modo que en ellos no hay elección.")


def t_calibracion_cal():
    """Umbral en el que cada mecanismo de los bloques canonicos rinde mejor."""
    BL = [("juniperus_seg_homogeneous_20260905_105741",   "\\textit{Data augment.} hom."),
          ("juniperus_seg_heterogeneous_20260905_111758", "\\textit{Data augment.} het."),
          ("juniperus_seg_by_size_20260830_202803",       "Tamaño"),
          ("juniperus_seg_by_region_20260831_073029",     "Región")]
    MEC = [("dense_baseline", "Modelo único"),
           ("btm_forest_ensemble_uniform", "\\textit{Ensemble} de expertos"),
           ("btm_forest_ensemble_weighted", "\\textit{Ens.} pesos por fiabilidad"),
           ("btm_forest_soup", "\\textit{Model merging}"),
           ("random_ensemble", "\\textit{Ensemble} no especializado")]
    filas = []
    for exp, be in BL:
        primero = True
        for mec, me in MEC:
            rt = _cal.optimo(exp, "test", mec)
            rf = _cal.optimo(exp, "field_work", mec)
            if rt is None and rf is None:
                continue
            filas.append([be if primero else "", me,
                          "--" if rt is None else f"{rt[1]:.2f}",
                          "--" if rt is None else f"{rt[0]:.2f}",
                          "--" if rf is None else f"{rf[1]:.2f}",
                          "--" if rf is None else f"{rf[0]:.2f}"])
            primero = False
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Umbral de confianza en el que cada mecanismo rinde mejor, y el F1 "
           "que alcanza en él, para los cuatro bloques que construyen expertos.}",
           "\\label{tab:exp6_calibracion}", "\\scriptsize",
           "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}ll@{\\hspace{8pt}}rr@{\\hspace{8pt}}rr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Bloque} & \\multirow{2}{*}{Mecanismo} & "
           "\\multicolumn{2}{c}{Test (PI)} & \\multicolumn{2}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
           " & & Umbral & F1 & Umbral & F1 \\\\", "\\midrule"]
    for f in filas:
        if f[0] and len(out) > 12:
            out.append("\\addlinespace[2pt]")
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def t_oe2_cal():
    """H2 con cada mecanismo en su umbral optimo."""
    BL = [("juniperus_seg_homogeneous_20260905_105741",   "\\textit{Data augmentation}, homogénea"),
          ("juniperus_seg_heterogeneous_20260905_111758", "\\textit{Data augmentation}, heterogénea"),
          ("juniperus_seg_by_size_20260830_202803",       "Tamaño del arbusto"),
          ("juniperus_seg_by_region_20260831_073029",     "Región geográfica")]
    filas = []
    for exp, nom in BL:
        fila = [nom]
        for split in ("test", "field_work"):
            u = _cal.optimo(exp, split, "dense_baseline")
            n = _cal.optimo(exp, split, "random_ensemble")
            e = _cal.optimo(exp, split, "btm_forest_ensemble_uniform")
            for r in (u, n, e):
                fila.append("--" if r is None else
                            f"{r[0]:.2f}")
            d = "--" if (n is None or e is None) else \
                ("$+" if e[0] - n[0] >= 0 else "$-") + f"{abs(e[0] - n[0]):.2f}$"
            fila.append(d)
        filas.append(fila)
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Los tres criterios de especialización resueltos con segmentación de "
           "instancias. Cada criterio "
           "con sus dos líneas base y con el \\textit{ensemble} de sus expertos, que es lo que "
           "la hipótesis H2 pone a prueba.}",
           "\\label{tab:exp8}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.9cm}"
           "@{\\hspace{6pt}}rrrr@{\\hspace{6pt}}rrrr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Criterio} & \\multicolumn{4}{c}{Test (PI)} & "
           "\\multicolumn{4}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           " & Único & \\makecell{No\\\\espec.} & Ens. & $\\Delta$ & Único & "
           "\\makecell{No\\\\espec.} & Ens. & $\\Delta$ \\\\",
           "\\midrule"]
    for f in filas:
        out.append(" & ".join(f) + " \\\\")
    out += ["\\midrule",
            "\\textit{Referencia} & \\multicolumn{4}{c}{\\textit{87.87}} & "
            "\\multicolumn{4}{c}{\\textit{76.86}} \\\\",
            "\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize $\\Delta$ es la diferencia entre el \\textit{ensemble} "
            "de expertos y el \\textit{ensemble} no especializado de su misma ejecución.",
            "\\end{table}"]
    return "\n".join(out)


def t_oe3_cal():
    """H3 con cada mecanismo en su umbral optimo, ordenado por acuerdo."""
    BL = [("juniperus_seg_by_size_20260828_072323", "Entrenamiento conjunto"),
          ("juniperus_seg_by_region_20260831_073029", "Región geográfica"),
          ("juniperus_seg_homogeneous_20260905_105741", "\\textit{Data augmentation}"),
          ("juniperus_seg_by_size_20260830_202803", "Tamaño del arbusto")]
    filas = []
    for exp, nom in BL:
        fila = [nom]
        for split in ("test", "field_work"):
            e = _cal.optimo(exp, split, "btm_forest_ensemble_uniform")
            f = _cal.optimo(exp, split, "btm_forest_soup")
            for r in (e, f):
                fila.append("--" if r is None else
                            f"{r[0]:.2f}")
            fila.append("--" if (e is None or f is None) else
                        ("$+" if f[0] - e[0] >= 0 else "$-") + f"{abs(f[0] - e[0]):.2f}$")
        filas.append(fila)
    return tabla(
        ["Conjunto de expertos", "\\textit{Ens.}", "Fus.", "$\\Delta$",
         "\\textit{Ens.}", "Fus.", "$\\Delta$"],
        filas, "@{}p{4.2cm}@{\\hspace{8pt}}rrr@{\\hspace{8pt}}rrr@{}",
        "El \\textit{model merging} frente al \\textit{ensemble} construido con esos "
        "mismos expertos, ordenados de mayor a menor "
        "parecido entre ellos.",
        "tab:oe3", small="script", tabcolsep="3pt",
        notas="$\\Delta$ es lo que gana o pierde el \\textit{model merging} respecto "
              "del \\textit{ensemble} construido con esos mismos expertos.")


def t_exp7_cal():
    """H4 con los tres mecanismos en su propio umbral optimo."""
    TAM = "juniperus_seg_by_size_20260830_202803"
    REFP = {"test": 87.87, "field_work": 76.86}
    FILAS = [(TAM, "dense_baseline", "Modelo único (ref. interna)"),
             ("copypaste_test", "dense_baseline", "\\textit{Copy-paste}"),
             ("distilled", "dense_baseline", "Destilación"),
             (TAM, "btm_forest_ensemble_uniform",
              "\\quad \\textit{Ensemble} por tamaño, su profesor")]
    filas, cortes = [], []
    for split, etiq in (("test", "Test (PI)"), ("field_work", "Campo (FW)")):
        if filas:
            cortes.append(len(filas))
        primero = True
        for exp, mec, nom in FILAS:
            r = _cal.optimo(exp, split, mec)
            if r is None:
                continue
            f1, thr, cota = r
            filas.append([etiq if primero else "", nom,
                          f"{thr:.2f}", f"{f1:.2f}",
                          ("$+" if f1 - REFP[split] >= 0 else "$-") +
                          f"{abs(f1 - REFP[split]):.2f}$"])
            primero = False
    return tabla(
        ["Conjunto", "Mecanismo", "Umbral", "Mejor F1", "vs. ref."],
        filas, "@{}llrrr@{}",
        "Los dos mecanismos de generación de señal y el modelo único del que "
        "parten. "
        "La última columna es la diferencia frente al trabajo previo "
        "(87{,}87 en test, 76{,}86 en campo).",
        "tab:exp7", small="script", tabcolsep="4pt", midrules=tuple(cortes),
        notas="El modelo único es la línea base con la que los dos mecanismos de "
              "esta sección comparten arquitectura y configuración de "
              "entrenamiento. La última fila es el \\textit{ensemble} de expertos "
              "por tamaño que genera las \\textit{pseudo-labels} de la destilación, "
              "y se da para poder medir el destilado también contra aquello que "
              "intenta comprimir.")


def t_condiciones_cal():
    """Efecto de filtrar las anotaciones, con cada mecanismo en su optimo."""
    CONJ = "juniperus_seg_by_size_20260828_072323"
    RANGO = "juniperus_seg_by_size_20260830_202803"
    MEC = [("expert_E_small_solo", "Experto pequeño (solo)"),
           ("expert_E_medium_solo", "Experto mediano (solo)"),
           ("expert_E_large_solo", "Experto grande (solo)"),
           ("btm_forest_ensemble_uniform", "\\textit{Ensemble} de expertos"),
           ("random_ensemble", "\\textit{Ensemble} no especializado"),
           ("dense_baseline", "Modelo único")]
    filas = []
    for mec, nom in MEC:
        fila = [nom]
        for exp in (CONJ, RANGO):
            for split in ("test", "field_work"):
                r = _cal.optimo(exp, split, mec)
                fila.append("--" if r is None else f"{r[0]:.2f}")
        filas.append(fila)
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Efecto de filtrar las anotaciones. En el entrenamiento conjunto los "
           "tres expertos ven todas las anotaciones y solo divergen en el \\textit{data "
           "augmentation}. En el entrenamiento por rango cada uno ve únicamente los "
           "arbustos de su tamaño.}",
           "\\label{tab:condiciones}", "\\small",
           "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}l@{\\hspace{8pt}}rr@{\\hspace{8pt}}rr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Mecanismo} & \\multicolumn{2}{c}{Entrenamiento conjunto} & "
           "\\multicolumn{2}{c}{Entrenamiento por rango} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           " & PI & FW & PI & FW \\\\", "\\midrule"]
    for k, f in enumerate(filas):
        if k == 3:
            out.append("\\midrule")
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize Referencia del trabajo previo, para las cuatro "
            "columnas, 87{,}87 en test y 76{,}86 en campo.",
            "\\end{table}"]
    return "\n".join(out)


# ================================================== resumen global (dashboard)
def t_resumen_global():
    d = carga_reeval()
    # una sola configuracion por experimento, la canonica de cada bloque
    FILAS = [
        ("legacy:mask2former_swinT",                    "Arquitectura \\textit{transformer} (cajas)"),
        ("juniperus_seg_homogeneous_20260905_105741",   "\\textit{Data augmentation}, homogénea"),
        ("juniperus_seg_heterogeneous_20260905_111758", "\\textit{Data augmentation}, heterogénea"),
        ("juniperus_seg_by_size_20260830_202803",       "Tamaño del arbusto"),
        ("juniperus_seg_by_region_20260831_073029",     "Región geográfica"),
        ("copypaste_test",                              "\\textit{Copy-paste}"),
        ("distilled",                                   "Destilación"),
    ]
    leg = pd.read_csv(A / "legacy_modelos.csv")
    leg = leg[(leg.metric == "S-IoU") & (leg.overlap == 50) & (leg.score_thr == 0.25)]
    MECS = ["btm_forest_ensemble_uniform", "btm_forest_soup", "random_ensemble"]
    UNICOS = ("legacy:mask2former_swinT", "copypaste_test", "distilled")
    filas = []
    for exp, nom in FILAS:
        fila = [nom]
        for split in ("test", "field_work"):
            if exp in UNICOS:
                if exp.startswith("legacy:"):
                    r = leg[(leg.model == exp.split(":", 1)[1]) & (leg.split == split)]
                else:
                    r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
                          (d.metric == "S-IoU") & (d.mechanism == "dense_baseline")]
                v = "--" if r.empty else f"{r.iloc[0].f1:.1f}"
                fila.append(f"\\multicolumn{{3}}{{c}}{{{v}}}")
                continue
            for m in MECS:
                r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
                      (d.metric == "S-IoU") & (d.mechanism == m)]
                fila.append("--" if r.empty else f"{r.iloc[0].f1:.1f}")
        filas.append(fila)

    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Resumen global. F1 vía S-IoU al 50\\,\\% de solapamiento y umbral "
           "de confianza 0.25, que es el valor por defecto del protocolo "
           "(Sección~\\ref{sec:protocolo}). Una configuración por experimento, la canónica de "
           "cada bloque. " + "}",
           "\\label{tab:resumen_global}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{2pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.4cm}"
           "@{\\hspace{8pt}}rrr@{\\hspace{8pt}}rrr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Experimento} & "
           "\\multicolumn{3}{c}{Test (PI)} & \\multicolumn{3}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           " & Ens. & Fus. & \\makecell{No\\\\espec.} & Ens. & Fus. & "
           "\\makecell{No\\\\espec.} \\\\",
           "\\midrule"]
    for f in filas:
        out.append(" & ".join(f) + " \\\\")
    out += ["\\midrule",
            "\\textit{Referencia} & "
            "\\multicolumn{3}{c}{\\textit{87.87}} & \\multicolumn{3}{c}{\\textit{76.86}} \\\\",
            "\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize «Ens.» es el \\textit{ensemble} de expertos con pesos "
            "uniformes, «Fus.» el \\textit{model merging} y «No espec.» el \\textit{ensemble} "
            "no especializado. Tres bloques no construyen expertos y son un solo modelo, cuya "
            "cifra ocupa las tres columnas. El de \\textit{transformers} se mide sobre "
            "\\textit{bounding boxes} y no sobre máscaras.",
            "\\end{table}"]
    return "\n".join(out)


def t_oe2():
    """Todos los resultados del sub-objetivo 2, tres criterios y segmentacion."""
    d = carga_reeval()
    FILAS = [
        ("juniperus_seg_homogeneous_20260905_105741",   "\\textit{Data augmentation}, homogénea"),
        ("juniperus_seg_heterogeneous_20260905_111758", "\\textit{Data augmentation}, heterogénea"),
        ("juniperus_seg_by_size_20260830_202803",       "Tamaño del arbusto"),
        ("juniperus_seg_by_region_20260831_073029",     "Región geográfica"),
    ]
    MECS = ["dense_baseline", "random_ensemble", "btm_forest_ensemble_uniform"]
    filas = []
    for exp, nom in FILAS:
        fila = [nom]
        for split in ("test", "field_work"):
            v = {}
            for m in MECS:
                r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
                      (d.metric == "S-IoU") & (d.mechanism == m)]
                v[m] = None if r.empty else float(r.iloc[0].f1)
            fila += [f"{v[m]:.2f}" if v[m] is not None else "--" for m in MECS]
            dif = v["btm_forest_ensemble_uniform"] - v["random_ensemble"]
            fila.append(("$+" if dif >= 0 else "$-") + f"{abs(dif):.2f}$")
        filas.append(fila)
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Los tres criterios de especialización resueltos con segmentación de "
           "instancias. Cada uno con sus dos líneas base y con el \\textit{ensemble} de sus "
           "expertos, que es lo que la hipótesis H2 pone a prueba. F1 vía S-IoU al 50\\,\\% de "
           "solapamiento y umbral de confianza 0{,}25. " + "}",
           "\\label{tab:exp8}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.9cm}"
           "@{\\hspace{6pt}}rrrr@{\\hspace{6pt}}rrrr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Criterio} & \\multicolumn{4}{c}{Test (PI)} & "
           "\\multicolumn{4}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           " & Único & \\makecell{No\\\\espec.} & Ens. & $\\Delta$ & Único & "
           "\\makecell{No\\\\espec.} & Ens. & $\\Delta$ \\\\",
           "\\midrule"]
    for f in filas:
        out.append(" & ".join(f) + " \\\\")
    out += ["\\midrule",
            "\\textit{Referencia} & \\multicolumn{4}{c}{\\textit{87.87}} & "
            "\\multicolumn{4}{c}{\\textit{76.86}} \\\\",
            "\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize «Único» es la línea base de modelo único y «No espec.» "
            "el \\textit{ensemble} no especializado, las dos referencias internas de cada "
            "criterio. «Ens.» es el \\textit{ensemble} de sus expertos y $\\Delta$ la "
            "diferencia frente al \\textit{ensemble} no especializado, que es la que mide la "
            "aportación de la especialización.",
            "\\end{table}"]
    return "\n".join(out)


def t_oe3():
    """Fusion de pesos frente al ensemble de los mismos expertos, por regimen."""
    import json
    d = carga_reeval()
    part = pd.read_csv(A / "participacion.csv")
    part = part[part.split == "test"]
    FILAS = [
        ("juniperus_seg_by_size_20260828_072323", "Expertos redundantes", 3),
        ("juniperus_seg_by_region_20260831_073029", "Región geográfica", 2),
        ("juniperus_seg_homogeneous_20260905_105741", "\\textit{Data augmentation}", 3),
        ("juniperus_seg_by_size_20260830_202803", "Tamaño del arbusto", 3),
    ]

    def v(exp, split, m):
        r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
              (d.metric == "S-IoU") & (d.mechanism == m)]
        return float(r.iloc[0].f1) if len(r) else None

    filas = []
    for exp, nom, n_exp in FILAS:
        r = part[part.experiment == exp]
        h = json.loads(r.iloc[0].agreement_histogram)
        ac = 100 * h[str(n_exp)] / r.iloc[0].total_gt_shrubs
        fila = [nom]
        for split in ("test", "field_work"):
            e = v(exp, split, "btm_forest_ensemble_uniform")
            f = v(exp, split, "btm_forest_soup")
            dif = f - e
            fila += [f"{e:.2f}", f"{f:.2f}",
                     ("$+" if dif >= 0 else "$-") + f"{abs(dif):.2f}$"]
        filas.append(fila)
    return tabla(
        ["Conjunto de expertos",
         "\\textit{Ens.}", "Fus.", "$\\Delta$", "\\textit{Ens.}", "Fus.", "$\\Delta$"],
        filas,
        "@{}p{4.2cm}@{\\hspace{8pt}}rrr@{\\hspace{8pt}}rrr@{}",
        "El \\textit{model merging} frente al \\textit{ensemble} construido con esos mismos "
        "expertos, ordenados de mayor a menor parecido entre ellos. F1 vía S-IoU al 50\\,\\% de "
        "solapamiento y umbral 0{,}25. ",
        "tab:oe3", small="script", tabcolsep="3pt",
        notas="«\\textit{Ens.}» es el \\textit{ensemble} de expertos y «Fus.» el "
              "\\textit{model merging}, en su versión de promedio simple. $\\Delta$ es lo "
              "que gana o pierde el \\textit{model merging} respecto del \\textit{ensemble} "
              "construido con esos mismos expertos, que es lo que la hipótesis H3 pone a prueba.")


def t_referencias():
    """Regla de seleccion de la canonica de cada bloque: la mejor sobre FW."""
    d = carga_reeval()
    leg = pd.read_csv(A / "legacy_modelos.csv")
    leg = leg[(leg.metric == "S-IoU") & (leg.overlap == 50) & (leg.score_thr == 0.25)]

    def f1(exp, split, mec):
        if exp.startswith("legacy:"):
            r = leg[(leg.model == exp.split(":", 1)[1]) & (leg.split == split)]
        else:
            r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
                  (d.metric == "S-IoU") & (d.mechanism == mec)]
        return None if r.empty else float(r.iloc[0].f1)

    UNI, ENS = "dense_baseline", "btm_forest_ensemble_uniform"
    A_UNI, A_ENS = "Único", "\\textit{Ens.}"
    BLOQUES = [
        ("Arquitectura \\textit{transformer}", A_UNI, UNI, [
            ("legacy:codetr_r50",        "Deformable DETR R50"),
            ("legacy:codetr_r50_aug",    "Deformable DETR R50 + \\textit{data augment.}"),
            ("legacy:codetr_r50_v1",     "Deformable DETR R50, config. inicial"),
            ("legacy:co_deform_swinL",   "Co-Deformable-DETR Swin-L"),
            ("legacy:co_dino_swinL",     "Co-DINO Swin-L"),
            ("legacy:co_dino_swinL_v1",  "Co-DINO Swin-L, v1"),
            ("legacy:mask2former_swinT", "Mask2Former Swin-T (cajas)"),
            ("legacy:mask2former_seg",   "Mask2Former Swin-T (máscaras)")]),
        ("\\textit{Data augment.}, homogénea", A_ENS, ENS, [
            ("juniperus_btm_homogeneous_20260824_103026", "Detección"),
            ("juniperus_seg_homogeneous_20260905_105741", "Segmentación")]),
        ("\\textit{Data augment.}, heterogénea", A_ENS, ENS, [
            ("juniperus_btm_heterogeneous_20260824_091248", "Detección"),
            ("juniperus_seg_heterogeneous_20260905_111758", "Segmentación")]),
        ("Tamaño del arbusto", A_ENS, ENS, [
            ("juniperus_seg_by_size_20260830_163740", "División natural, 50 ép."),
            ("juniperus_seg_by_size_20260830_164608", "Cantidad igualada, 50 ép."),
            ("juniperus_seg_by_size_20260830_202803", "División natural, 100 ép."),
            ("juniperus_seg_by_size_20260830_221329", "Modelo grande, 100 ép.")]),
        ("Región geográfica", A_ENS, ENS, [
            ("juniperus_seg_by_region_20260828_114011", "Primera ejecución"),
            ("juniperus_seg_by_region_20260831_073029", "Ejecución canónica")]),
        ("\\textit{Copy-paste}", A_UNI, UNI, [
            ("copypaste_test", "Modelo único aumentado")]),
        ("Destilación", A_UNI, UNI, [
            ("distilled", "Estudiante del \\textit{ensemble}")]),
    ]

    def celda(cand):
        vals = [(nom, v) for nom, v in cand if v is not None]
        if not vals:
            return "--"
        nom, v = max(vals, key=lambda c: c[1])
        return nom + ", " + f"{v:.2f}"

    filas = []
    for bloque, arb, mec, variantes in BLOQUES:
        pi = [(nom, f1(e, "test", mec)) for e, nom in variantes]
        fw = [(nom, f1(e, "field_work", mec)) for e, nom in variantes]
        filas.append([bloque, arb, celda(pi),
                      "\\textbf{" + celda(fw) + "}"])

    return tabla(
        ["Bloque", "Árbitro", "Mejor sobre PI",
         "Mejor sobre FW, que es la canónica"],
        filas,
        "@{}>{\\raggedright\\arraybackslash}p{2.7cm}"
        ">{\\raggedright\\arraybackslash}p{1.2cm}"
        ">{\\raggedright\\arraybackslash}p{3.8cm}"
        ">{\\raggedright\\arraybackslash}p{4.0cm}@{}",
        "Configuración canónica de cada bloque. Para cada uno se da la variante "
        "que mejor rinde sobre el test fotointerpretado y la que mejor rinde "
        "sobre el test de trabajo de campo, medidas sobre el mecanismo que "
        "contrasta la hipótesis del bloque. F1 vía S-IoU al 50\\,\\% de "
        "solapamiento y umbral de confianza 0{,}25.",
        "tab:referencias", small="script", tabcolsep="3pt",
        notas="«Único» es la línea base de modelo único y «\\textit{Ens.}» el "
              "\\textit{ensemble} de expertos. Los dos bloques de ampliación de la señal tienen una sola "
              "configuración cada uno, de modo que en ellos no hay elección. "
              "Una configuración sin medida sobre el test de trabajo de campo no "
              "puede ser canónica, lo que deja fuera a tres configuraciones del "
              "bloque de arquitecturas.")


def t_variantes():
    """Anexo: todas las variantes de cada bloque, no solo la canonica."""
    d = carga_reeval()
    leg = pd.read_csv(A / "legacy_modelos.csv")
    leg = leg[(leg.metric == "S-IoU") & (leg.overlap == 50) & (leg.score_thr == 0.25)]
    FILAS = [
        (None, "\\textit{Arquitecturas transformer} (cajas)"),
        ("legacy:codetr_r50",        "\\quad Deformable DETR R50, 1200 ép."),
        ("legacy:codetr_r50_aug",    "\\quad Deformable DETR R50 + \\textit{data augment.}"),
        ("legacy:codetr_r50_v1",     "\\quad Deformable DETR R50, inicial"),
        ("legacy:co_deform_swinL",   "\\quad Co-Deformable-DETR Swin-L"),
        ("legacy:co_dino_swinL",     "\\quad Co-DINO Swin-L"),
        ("legacy:co_dino_swinL_v1",  "\\quad Co-DINO Swin-L, v1"),
        ("legacy:mask2former_swinT", "\\quad Mask2Former Swin-T (cajas)"),
        ("legacy:mask2former_seg",   "\\quad Mask2Former Swin-T (máscaras)$^{*}$"),
        (None, "\\textit{Data augmentation}"),
        ("juniperus_btm_homogeneous_20260824_103026",   "\\quad Homogénea, detección (cajas)"),
        ("juniperus_btm_heterogeneous_20260824_091248", "\\quad Heterogénea, detección (cajas)"),
        ("juniperus_seg_homogeneous_20260905_105741",   "\\quad Homogénea, segmentación$^{*}$"),
        ("juniperus_seg_heterogeneous_20260905_111758", "\\quad Heterogénea, segmentación$^{*}$"),
        (None, "\\textit{Tamaño del arbusto}"),
        ("juniperus_seg_by_size_20260830_163740", "\\quad División natural, 50 ép."),
        ("juniperus_seg_by_size_20260830_164608", "\\quad Cantidad igualada, 50 ép."),
        ("juniperus_seg_by_size_20260830_202803", "\\quad División natural, 100 ép.$^{*}$"),
        ("juniperus_seg_by_size_20260830_221329", "\\quad Modelo grande, 100 ép."),
        (None, "\\textit{Región geográfica}"),
        ("juniperus_seg_by_region_20260828_114011", "\\quad Primera ejecución"),
        ("juniperus_seg_by_region_20260831_073029", "\\quad Ejecución canónica$^{*}$"),
        (None, "\\textit{Ampliación de la señal}"),
        ("copypaste_test", "\\quad \\textit{Copy-paste}$^{*}$"),
        ("distilled",      "\\quad Destilación$^{*}$"),
        (None, "\\textit{Condición de entrenamiento conjunto}"),
        ("juniperus_seg_by_size_20260828_072323", "\\quad Sin filtrar anotaciones"),
    ]
    MECS = ["dense_baseline", "btm_forest_ensemble_uniform", "btm_forest_soup",
            "random_ensemble"]
    filas, seps = [], []
    for exp, nom in FILAS:
        if exp is None:
            seps.append(len(filas))
            filas.append([f"\\multicolumn{{9}}{{@{{}}l}}{{{nom}}}"])
            continue
        fila = [nom]
        for split in ("test", "field_work"):
            if exp.startswith("legacy:"):
                fila += [_cal_celda(exp, split, "dense_baseline", dec=2), "--", "--", "--"]
                continue
            for m in MECS:
                fila.append(_cal_celda(exp, split, m, dec=2))
        filas.append(fila)

    out = ["\\begin{table}[htbp]", "\\centering",
           "\\caption{Todas las configuraciones evaluadas de cada bloque. Las "
           "marcadas con $^{*}$ son las canónicas, es decir, las que entran en la "
           "comparación global de la Sección~\\ref{sec:global}.}",
           "\\label{tab:variantes}", "\\scriptsize",
           "\\setlength{\\tabcolsep}{1.5pt}",
           "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.1cm}"
           "@{\\hspace{3pt}}rrrr@{\\hspace{3pt}}rrrr@{}}",
           "\\toprule",
           "\\multirow{2}{*}{Variante} & \\multicolumn{4}{c}{Test (PI)} & "
           "\\multicolumn{4}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           " & Único & Ens. & Fus. & \\makecell{No\\\\espec.} & Único & Ens. & Fus. & "
           "\\makecell{No\\\\espec.} \\\\",
           "\\midrule"]
    for k, f in enumerate(filas):
        if k in seps and k:
            out.append("\\addlinespace[2pt]")
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize La variante de modelo grande emplea el reparto por "
            "cantidad igualada. Las filas marcadas «(cajas)» se miden sobre "
            "\\textit{bounding boxes} y el resto sobre máscaras.",
            "\\end{table}"]
    return "\n".join(out)


# ============================================================== SAHI y otros
# Los JSON originales de sahi_results se recuperaron del volcado del servidor
# (analisis/servidor/asantana/model_soup/sahi_results/) el 6 de septiembre.
# SAHI_RESPALDO se conserva como red de seguridad: si esa ruta desaparece,
# reproduce exactamente los mismos valores (verificado bit a bit).
SAHI_RESPALDO = [
    ["0.25", "--",    "67.45", "62.06", "$-5.38$",  "79.26", "64.14", "$-15.12$"],
    ["0.40", "15 px", "67.46", "69.18", "$+1.72$",  "75.36", "69.41", "$-5.95$"],
    ["0.40", "5 px",  "67.46", "69.94", "$+2.48$",  "75.36", "70.25", "$-5.11$"],
]


def t_sahi():
    import glob as _g
    NOM = {"sahi_dense_baseline_field_work_tile224_ov0.25.json": ("0.25", "--"),
           "sahi_dense_baseline_field_work_tile224_ov0.25_conf0.4_edge15.json": ("0.40", "15 px"),
           "sahi_dense_baseline_field_work_tile224_ov0.25_conf0.4_edge5.json": ("0.40", "5 px")}
    SRV = "/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup"
    filas = []
    for f in sorted(_g.glob(f"{SRV}/sahi_results/*.json")):
        base = f.split("/")[-1]
        if base not in NOM:
            continue
        conf, borde = NOM[base]
        j = json.load(open(f))
        b, t = j["baseline"], j["tiled"]
        filas.append([conf, borde,
                      f"{100*b['iou']['f1']:.2f}", f"{100*t['iou']['f1']:.2f}",
                      f"${100*(t['iou']['f1']-b['iou']['f1']):+.2f}$",
                      f"{100*b['s_iou']['f1']:.2f}", f"{100*t['s_iou']['f1']:.2f}",
                      f"${100*(t['s_iou']['f1']-b['s_iou']['f1']):+.2f}$"])
    filas.sort(key=lambda r: (r[0], r[1]))
    if not filas:
        filas = [list(f) for f in SAHI_RESPALDO]
    return tabla(
        ["Conf.", "Margen", "IoU sin", "IoU con", "$\\Delta$",
         "S-IoU sin", "S-IoU con", "$\\Delta$"],
        filas, "@{}llrrrrrr@{}",
        "\\textit{Slicing}. Efecto sobre el test de trabajo de campo. La columna «sin» "
        "corresponde a la evaluación sobre la imagen completa y la columna «con» a "
        "la evaluación con \\textit{slicing}.",
        "tab:exp5_sahi", tabcolsep="4pt",
        notas="Recortes de $224\\times224$ px con 25\\,\\% de solapamiento. «Margen» es "
              "la anchura del filtro de borde interno de recorte.")


def t_por_bucket():
    d = pd.read_csv(A / "por_bucket.csv")
    MECS = ["dense_baseline", "expert_E_small_solo", "expert_E_medium_solo",
            "expert_E_large_solo", "btm_forest_ensemble_uniform", "random_ensemble"]
    filas = []
    for m in MECS:
        fila = [NOMBRE_MEC.get(m, m)]
        for split in ("test", "field_work"):
            for b in ("small", "medium", "large"):
                r = d[(d.variante == m) & (d.split == split) &
                      (d.bucket == b) & (d.metrica == "S-IoU")]
                fila.append("--" if r.empty else f"{r.iloc[0].f1:.1f}")
        filas.append(fila)
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Especialización por tamaño. F1 vía S-IoU desglosado por el tamaño real del "
           "arbusto anotado. Los verdaderos positivos y los falsos negativos se atribuyen "
           "al rango del objeto anotado. Los falsos positivos se atribuyen al rango de la "
           "propia predicción. " + "}",
           "\\label{tab:por_bucket}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}l@{\\hspace{6pt}}rrr@{\\hspace{6pt}}rrr@{}}", "\\toprule",
           "\\multirow{2}{*}{Mecanismo} & \\multicolumn{3}{c}{Test (PI)} & "
           "\\multicolumn{3}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           " & Peq. & Med. & Gra. & Peq. & Med. & Gra. \\\\", "\\midrule"]
    for f in filas:
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def t_calibracion():
    d = pd.read_csv(A / "barrido_confianza.csv")
    MECS = ["dense_baseline", "btm_forest_ensemble_uniform", "btm_forest_soup",
            "random_ensemble"]
    filas = []
    corte = None
    for split, etiq in (("test", "Test (PI)"), ("field_work", "Campo (FW)")):
        for m in MECS:
            s_ = d[(d.split == split) & (d.mechanism == m) & (d.metric == "S-IoU")]
            if s_.empty:
                continue
            piso = float(s_.iloc[0].score_min_exportado)
            # solo tienen sentido los umbrales por encima del corte de exportacion
            s_ = s_[s_.score_thr >= piso - 0.006]
            base = s_[s_.score_thr == 0.25]
            i = s_["f1"].idxmax()
            opt = s_.loc[i, "score_thr"]
            en_el_piso = opt <= s_.score_thr.min() + 1e-9 and piso > 0.02
            marca = "$\\leq$" if en_el_piso else ""
            b = base.iloc[0].f1 if not base.empty else float("nan")
            filas.append([etiq if m == MECS[0] else "", NOMBRE_MEC.get(m, m),
                          "--" if base.empty else f"{b:.2f}",
                          f"{s_.loc[i,'f1']:.2f}", f"{marca}{opt:.2f}",
                          f"${s_.loc[i,'f1'] - b:+.2f}$"])
        if split == "test":
            corte = len(filas)
    return tabla(
        ["Conjunto", "Mecanismo", "F1 a 0.25", "Mejor F1", "Umbral óptimo", "$\\Delta$"],
        filas, "@{}lllrrr@{}",
        "Calibración del umbral. Cada mecanismo evaluado en su mejor umbral de confianza "
        "frente al umbral estándar. ",
        "tab:exp6_calibracion", small="script", tabcolsep="4pt",
        midrules=(corte,),
        notas="El símbolo $\\leq$ indica que el óptimo coincide con el umbral más bajo "
              "que las predicciones exportadas permiten explorar, de modo que el valor "
              "es una cota inferior. El \\textit{model merging} se reexportó después "
              "desde 0.02 sobre los dos conjuntos. Los otros tres mecanismos conservan "
              "sobre el test de trabajo de campo sus predicciones filtradas a 0.25, de "
              "modo que ahí no puede explorarse por debajo de ese valor.")


def t_legacy_siou():
    """Modelos clasicos evaluados con las metricas del paper (geometria de caja)."""
    d = pd.read_csv(A / "legacy_modelos.csv")
    try:
        y = pd.read_csv(A / "yolo_cajas.csv")
    except FileNotFoundError:
        y = None
    ORDEN = ["codetr_r50", "codetr_r50_aug", "co_deform_swinL", "co_dino_swinL",
             "mask2former_swinT"]
    ETIQ = {"codetr_r50": "Deformable DETR R50 (1200 ép.)",
            "codetr_r50_aug": "Deformable DETR R50 + \\textit{data augment.}",
            "co_deform_swinL": "Co-DETR (Co-Deformable) Swin-L",
            "co_dino_swinL": "Co-DETR (Co-DINO) Swin-L",
            "mask2former_swinT": "Mask2Former Swin-T"}

    def mejor(df, split, modelo):
        ss = df[(df.split == split) & (df.model == modelo) & (df.overlap == 50) &
                (df.metric == "S-IoU")]
        si = df[(df.split == split) & (df.model == modelo) & (df.overlap == 50) &
                (df.metric == "IoU")]
        if ss.empty:
            return None
        # El umbral se elige maximizando el F1 con S-IoU, que es la metrica
        # primaria, y TODAS las columnas de la fila se leen en ese mismo umbral,
        # de modo que la fila describe un unico punto de operacion.
        i = ss["f1"].idxmax()
        thr = ss.loc[i, "score_thr"]
        sij = si[si.score_thr == thr]
        iou_f1 = f"{sij['f1'].iloc[0]:.2f}" if len(sij) else "--"
        return [f"{thr:.2f}", iou_f1,
                f"{ss.loc[i,'precision']:.1f}", f"{ss.loc[i,'recall']:.1f}",
                f"{ss.loc[i,'f1']:.2f}"]

    filas = [["\\textit{Referencia}", "\\textit{Mask R-CNN (máscaras)}", "--",
              "\\textit{84.84}", "\\textit{88.55}", "\\textit{87.20}", "\\textit{87.87}"]]
    cortes = [1]
    for split, etiq in (("test", "Test (PI)"), ("field_work", "Campo (FW)")):
        primero = True
        for m in ORDEN:
            r = mejor(d, split, m)
            if r is None:
                continue
            filas.append([etiq if primero else "", ETIQ[m]] + r)
            primero = False
        if y is not None:
            r = mejor(y, split, "yolo_dense_baseline")
            if r is not None:
                filas.append(["" if not primero else etiq,
                              "\\textit{YOLOv8n-seg único (cajas)}",
                              r[0], r[1], r[2], r[3], f"\\textit{{{r[4]}}}"])
        if split == "test":
            cortes.append(len(filas))
    return tabla(
        ["Conjunto", "Modelo", "Conf.", "IoU F1", "S-IoU P", "S-IoU R", "S-IoU F1"],
        filas, "@{}lllrrrr@{}",
        "Configuraciones de la familia \\textit{transformer} evaluadas con las "
        "métricas del trabajo de referencia. Las cuatro columnas de métrica de "
        "cada fila corresponden al mismo umbral de confianza.",
        "tab:legacy_siou", small="script", tabcolsep="3pt", midrules=tuple(cortes),
        notas="Todas las métricas de este cuadro se calculan sobre "
              "\\textit{bounding boxes} y la referencia emplea polígonos "
              "(Sección~\\ref{sec:comparabilidad}). La fila en cursiva de cada bloque es el "
              "modelo ligero de segmentación de la especialización por tamaño, reevaluado "
              "también sobre \\textit{bounding boxes} para poder compararlo en igualdad de "
              "condiciones.")



def t_paper_original():
    """Tabla 4 del trabajo de referencia, en su formato original."""
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Resultado del trabajo de referencia \\cite{KHALDI2024104191}: "
           "Mask R-CNN evaluada sobre el test fotointerpretado (PI, con "
           "$\\theta_{score}=90\\%$) y el test de trabajo de campo (FW, con "
           "$\\theta_{score}=50\\%$), con las métricas IoU y S-IoU a los umbrales "
           "de solapamiento del 50\\,\\% y el 75\\,\\%.}",
           "\\label{tab:paper_original}", "\\small",
           "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}llcrrrrrr@{}}", "\\toprule",
           "Conjunto & Métrica & Umbral & TP & FP & FN & Precisión & Exhaust. & F1 \\\\",
           "\\midrule"]
    for si, (split, etiq) in enumerate([("test", "Test (PI)"), ("field_work", "Campo (FW)")]):
        if si:
            out.append("\\midrule")
        primero = True
        for met in ("IoU", "S-IoU"):
            for thr in (50, 75):
                tp, fp, fn = PAPER_TPFPFN[(split, met, thr)]
                p_, r_, f1v = PAPER[(split, met, thr)]
                celda = ("\\multirow{4}{*}{" + etiq + "}") if primero else ""
                primero = False
                out.append(f"{celda} & {met} & {thr}\\% & {tp} & {fp} & {fn} & "
                           f"{p_:.2f} & {r_:.2f} & {f1v:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def t_sintesis():
    filas = [
        ["Modelo único", "\\makecell[l]{¿Puede una arquitectura\\\\mejor superar a la referencia?}",
         "\\makecell[l]{En validación sí, en campo no.\\\\Domina el extractor}",
         "92,0\\,\\% mAP@50"],
        ["\\textit{Data augment.}", "\\makecell[l]{¿Sirve la divergencia\\\\inducida por el aumento?}",
         "\\makecell[l]{No. Los expertos resuelven\\\\el mismo problema}",
         "$-1{,}0$ vs. no espec."],
        ["Tamaño", "\\makecell[l]{¿Y un eje real que reparta\\\\las anotaciones?}",
         "\\makecell[l]{Sí. Acuerdo entre expertos\\\\prácticamente nulo}",
         "$+11{,}7$ vs. no espec."],
        ["Geografía", "\\makecell[l]{¿Basta con que el eje\\\\sea real?}",
         "\\makecell[l]{No. Repartir escenas deja\\\\un 80\\,\\% de acuerdo}",
         "$+2{,}9$ vs. no espec."],
        ["\\textit{Slicing}", "\\makecell[l]{¿Ayuda subir la resolución\\\\de la imagen en inferencia?}",
         "\\makecell[l]{Mixto. Mejora IoU y empeora\\\\S-IoU, la métrica que cuenta}",
         "$-5{,}1$ en S-IoU"],
        ["Umbral", "\\makecell[l]{¿Es justo un umbral\\\\de confianza común?}",
         "\\makecell[l]{No. Cada mecanismo y cada\\\\conjunto tiene su óptimo}",
         "$+31$ al recalibrar"],
        ["Más señal", "\\makecell[l]{¿Se puede generar señal\\\\sin datos nuevos?}",
         "\\makecell[l]{Sí, copiando y pegando.\\\\Destilar no funciona}",
         "$+2{,}65$ vs. referencia"],
    ]
    return tabla(
        ["Bloque", "Pregunta", "Respuesta", "Cifra clave"],
        filas, "@{}clll@{}",
        "Síntesis de los siete bloques experimentales, con la pregunta que responde "
        "cada uno, la respuesta obtenida y la cifra que la sostiene. Las cifras de los "
        "tres ejes de especialización son la diferencia en F1 vía S-IoU entre el "
        "\\textit{ensemble} de expertos y su propio \\textit{ensemble} no especializado "
        "sobre el test de trabajo de campo.",
        "tab:sintesis", small="script", tabcolsep="2pt")



def t_modos_fallo():
    """Desglose de como se resuelve cada anotacion y cada prediccion."""
    d = pd.read_csv(A / "casos.csv")
    g, pr = d[d.tipo == "gt"], d[d.tipo == "pred"]
    filas, col = [], {}
    for sp in ("test", "field_work"):
        s_, q = g[g.split == sp], pr[pr.split == sp]
        n, m = len(s_), len(q)
        col[sp] = dict(
            n=n,
            unica=((s_.detectada) & (s_.n_solapan == 1)).sum(),
            frag=((s_.detectada) & (s_.n_solapan >= 2)).sum(),
            nodet=(~s_.detectada).sum(),
            nodet_peq=((~s_.detectada) & (s_.bucket == "small")).sum(),
            m=m,
            fp=(q.n_solapan == 0).sum(),
            fus=(q.n_solapan >= 2).sum())

    def par(k, base):
        a, b = col["test"], col["field_work"]
        return [f"{a[k]}", f"{100*a[k]/a[base]:.1f}", f"{b[k]}", f"{100*b[k]/b[base]:.1f}"]

    filas.append(["\\textit{Anotaciones evaluadas}",
                  f"\\textit{{{col['test']['n']}}}", "", f"\\textit{{{col['field_work']['n']}}}", ""])
    filas.append(["Detectada por una sola predicción"] + par("unica", "n"))
    filas.append(["Detectada de forma fragmentada"] + par("frag", "n"))
    filas.append(["No detectada"] + par("nodet", "n"))
    filas.append(["\\quad de ellas, del rango pequeño"] + par("nodet_peq", "n"))
    filas.append(["\\textit{Predicciones emitidas}",
                  f"\\textit{{{col['test']['m']}}}", "", f"\\textit{{{col['field_work']['m']}}}", ""])
    filas.append(["Sin ninguna anotación debajo"] + par("fp", "m"))
    filas.append(["Agrupa dos o más anotaciones"] + par("fus", "m"))

    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Cómo se resuelve cada anotación y cada predicción del modelo único, "
           "evaluado en el umbral en el que mejor rinde sobre cada conjunto y con "
           "solapamiento del 50\\,\\%. Los porcentajes son "
           "sobre el total de anotaciones o de predicciones de cada conjunto. " + "}",
           "\\label{tab:modos_fallo}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}l@{\\hspace{8pt}}rr@{\\hspace{8pt}}rr@{}}", "\\toprule",
           "\\multirow{2}{*}{} & \\multicolumn{2}{c}{Test (PI)} & \\multicolumn{2}{c}{Campo (FW)} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           " & n & \\% & n & \\% \\\\", "\\midrule"]
    for i, f in enumerate(filas):
        if i == 5:
            out.append("\\midrule")
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)



def t_condiciones():
    """Expertos entrenados sobre el conjunto completo vs. sobre su rango."""
    d = carga_reeval()
    A_ = "juniperus_seg_by_size_20260828_072323"   # cada experto ve todas las anotaciones
    B_ = "juniperus_seg_by_size_20260830_202803"   # cada experto ve solo su rango
    MECS = [("expert_E_small_solo", "Experto pequeño (solo)"),
            ("expert_E_medium_solo", "Experto mediano (solo)"),
            ("expert_E_large_solo", "Experto grande (solo)"),
            ("btm_forest_ensemble_uniform", "\\textit{Ensemble} de expertos"),
            ("random_ensemble", "\\textit{Ensemble} no especializado"),
            ("dense_baseline", "Modelo único")]

    def v(exp, split, m):
        r = d[(d.experiment == exp) & (d.split == split) & (d.overlap == 50) &
              (d.metric == "S-IoU") & (d.mechanism == m)]
        return f"{r.iloc[0].f1:.2f}" if len(r) else "--"

    filas = [[nom, v(A_, "test", m), v(A_, "field_work", m),
              v(B_, "test", m), v(B_, "field_work", m)] for m, nom in MECS]
    out = ["\\begin{table}[H]", "\\centering",
           "\\caption{Efecto de filtrar las anotaciones. En el entrenamiento conjunto "
           "los tres expertos ven todas las anotaciones y solo divergen en el "
           "\\textit{data augmentation}. En el entrenamiento por rango cada uno ve "
           "únicamente los arbustos de su tamaño. F1 vía S-IoU al 50\\,\\% de "
           "solapamiento.}",
           "\\label{tab:condiciones}", "\\small", "\\setlength{\\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}l@{\\hspace{8pt}}rr@{\\hspace{8pt}}rr@{}}", "\\toprule",
           "\\multirow{2}{*}{Mecanismo} & \\multicolumn{2}{c}{Entrenamiento conjunto} & "
           "\\multicolumn{2}{c}{Entrenamiento por rango} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
           " & PI & FW & PI & FW \\\\", "\\midrule"]
    for i, f in enumerate(filas):
        if i == 3:
            out.append("\\midrule")
        out.append(" & ".join(f) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            "\\\\[2pt]\\footnotesize Referencia del trabajo previo, para las cuatro "
            "columnas, 87{,}87 en test y 76{,}86 en campo.",
            "\\end{table}"]
    return "\n".join(out)



def t_exp7():
    """Los dos mecanismos que generan mas senal, cada uno en su mejor punto."""
    d = pd.read_csv(A / "exp7_barrido.csv")
    r = carga_reeval()
    EXP_DENSO = "juniperus_seg_by_size_20260830_202803"
    PAPER_F1 = {"test": 87.87, "field_work": 76.86}
    filas, corte = [], None
    for split, etiq in (("test", "Test (PI)"), ("field_work", "Campo (FW)")):
        primero = True
        # referencia del modelo denso, al mismo umbral estandar
        dn = r[(r.experiment == EXP_DENSO) & (r.split == split) & (r.overlap == 50) &
               (r.metric == "S-IoU") & (r.mechanism == "dense_baseline")]
        if len(dn):
            f = float(dn.iloc[0].f1)
            filas.append([etiq, "Modelo único (ref. interna)", "0.25",
                          f"{f:.2f}", f"${f - PAPER_F1[split]:+.2f}$"])
            primero = False
        for m, nom in (("copy-paste", "\\textit{Copy-paste}"), ("destilado", "Destilación")):
            s_ = d[(d.split == split) & (d.mechanism == m) & (d.metric == "S-IoU") &
                   (d.overlap == 50)]
            if s_.empty:
                continue
            i = s_["f1"].idxmax()
            base = s_[s_.score_thr == 0.25]
            filas.append([etiq if primero else "", nom,
                          f"{s_.loc[i,'score_thr']:.2f}",
                          f"{s_.loc[i,'f1']:.2f}",
                          f"${s_.loc[i,'f1'] - PAPER_F1[split]:+.2f}$"])
            primero = False
        if split == "test":
            corte = len(filas)
    return tabla(
        ["Conjunto", "Mecanismo", "Umbral", "Mejor F1", "vs. ref."],
        filas, "@{}llrrr@{}",
        "Los dos mecanismos de generación de señal, cada uno evaluado en el umbral "
        "de confianza que maximiza su F1. La última columna es la diferencia frente "
        "al trabajo previo (87{,}87 en test, 76{,}86 en campo). F1 vía S-IoU al 50\\,\\% "
        "de solapamiento. ",
        "tab:exp7", small="script", tabcolsep="4pt", midrules=(corte,),
        notas="El modelo único se incluye al umbral estándar como referencia interna, "
              "por ser la línea base con la que los dos mecanismos de esta sección "
              "comparten arquitectura y configuración de entrenamiento.")


# =========================================================================
if __name__ == "__main__":
    import sys
    GEN = {
        "tabla_inventario":      t_inventario,
        "tabla_modos_fallo":     t_modos_fallo,
        "tabla_condiciones":     t_condiciones_cal,
        "tabla_exp7":            t_exp7_cal,
        "tabla_sintesis":        t_sintesis,
        "tabla_paper_original":  t_paper_original,
        "tabla_legacy":          t_legacy,
        "tabla_legacy_siou":     t_legacy_siou,
        "tabla_sahi":            t_sahi,
        "tabla_por_bucket":      t_por_bucket,
        "tabla_calibracion":     t_calibracion_cal,
        "tabla_resumen_global":  t_resumen_global_cal,
        "tabla_exp2_test": lambda: t_experimento(
            "juniperus_btm_heterogeneous_20260824_091248", "test",
            ["dense_baseline", "expert_E1_scale_solo", "expert_E2_illum_solo",
             "expert_E3_geom_solo", "btm_forest_ensemble_uniform", "random_ensemble"],
            "Divergencia inducida, variante heterogénea, test fotointerpretado. Solapamiento "
            "del 50\\%. Métricas sobre \\textit{bounding boxes}.", "tab:exp2_test", False),
        "tabla_exp2_fw": lambda: t_experimento(
            "juniperus_btm_heterogeneous_20260824_091248", "field_work",
            ["dense_baseline", "expert_E1_scale_solo", "expert_E2_illum_solo",
             "expert_E3_geom_solo", "btm_forest_ensemble_uniform", "random_ensemble"],
            "Divergencia inducida, variante heterogénea, test de trabajo de campo. Solapamiento "
            "del 50\\%. Métricas sobre \\textit{bounding boxes}.", "tab:exp2_fw", False),
        "tabla_exp2_hom": lambda: t_experimento(
            "juniperus_btm_homogeneous_20260824_103026", "test",
            ["dense_baseline", "btm_forest_ensemble_uniform", "btm_forest_soup",
             "random_ensemble"],
            "Divergencia inducida, variante homogénea, test fotointerpretado. Solapamiento "
            "del 50\\%.", "tab:exp2_hom", False),
        "tabla_exp3_test": lambda: t_experimento_cal(
            "juniperus_seg_by_size_20260830_202803", "test",
            ["dense_baseline", "expert_E_small_solo", "expert_E_medium_solo",
             "expert_E_large_solo", "btm_forest_ensemble_uniform",
             "btm_forest_soup", "random_ensemble"],
            "Especialización por tamaño, test fotointerpretado (690 arbustos).",
            "tab:exp3_test"),
                "tabla_exp3_fw": lambda: t_experimento_cal(
            "juniperus_seg_by_size_20260830_202803", "field_work",
            ["dense_baseline", "expert_E_small_solo", "expert_E_medium_solo",
             "expert_E_large_solo", "btm_forest_ensemble_uniform",
             "btm_forest_soup", "random_ensemble"],
            "Especialización por tamaño, test de trabajo de campo (1.771 arbustos).",
            "tab:exp3_fw"),
                "tabla_exp4_test": lambda: t_experimento_cal(
            "juniperus_seg_by_region_20260831_073029", "test",
            ["dense_baseline", "expert_E_region0_solo", "expert_E_region1_solo",
             "btm_forest_ensemble_uniform", "btm_forest_soup", "random_ensemble"],
            "Especialización geográfica, test fotointerpretado.",
            "tab:exp4_test"),
                "tabla_exp4_fw": lambda: t_experimento_cal(
            "juniperus_seg_by_region_20260831_073029", "field_work",
            ["dense_baseline", "expert_E_region0_solo", "expert_E_region1_solo",
             "btm_forest_ensemble_uniform", "btm_forest_soup", "random_ensemble"],
            "Especialización geográfica, test de trabajo de campo.",
            "tab:exp4_fw"),
                "tabla_oe2": lambda: t_oe2_cal(),
        "tabla_oe3": lambda: t_oe3_cal(),
        "tabla_referencias": lambda: t_referencias_cal(),
        "tabla_variantes": lambda: t_variantes(),
        "tabla_comparativa": lambda: t_comparativa(
            "juniperus_seg_by_size_20260830_202803",
            ["dense_baseline", "btm_forest_ensemble_uniform"],
            "Comparación con el trabajo de referencia, en su mismo formato, con "
            "cada conjunto desglosado por métrica de solapamiento y umbral. "
            "Todos los valores en porcentaje. ",
            "tab:comparativa_paper"),
    }
    pedidas = sys.argv[1:] or list(GEN)
    for n in pedidas:
        try:
            escribe(n, GEN[n]())
        except Exception as e:
            print(f"  [!] {n}: {type(e).__name__}: {e}")
