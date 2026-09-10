#!/usr/bin/env python3
"""Genera las cuatro figuras de ejemplos reales que acompañan la discusión de resultados.

Lee las predicciones guardadas de Mask2Former, de los expertos por tamaño y del ensemble
heterogéneo, y selecciona automáticamente la escena de mayor contraste en cada conjunto de
evaluación. Escribe en PNG ejemplo_geometria.png, la misma predicción como máscara y como caja
envolvente; ejemplo_umbral.png, las detecciones que retira el umbral óptimo; ejemplo_expertos.png,
lo que detecta cada experto; y ejemplo_merging.png, el ensemble frente al model merging.
"""
import sys, json
sys.path.insert(0, "/home/santana/Documents/docs_TFM/analisis")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from estilo import TEXTWIDTH, FIGS


def guardar_png(fig, nombre, dpi=200):
    """Estas cuatro figuras son casi todo imagen, de modo que en PDF vectorial
    pesan mucho sin ganar nada. Se guardan en PNG."""
    out = FIGS / f"{nombre}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print("  ->", out.name)

IMG = {"test": "/home/santana/Documents/TFM/Photo_Interpretation_Data_jpg/Test/Images",
       "field_work": "/home/santana/Documents/TFM/Field_Work_Data_jpg/External_Val_Data/Images"}
SRV = "/home/santana/Documents/docs_TFM/analisis/servidor/asantana/model_soup/experiments"
EXP8 = "/home/santana/Documents/docs_TFM/analisis/exp8/experiments"
TAM = "juniperus_seg_by_size_20260830_202803"
HET = "juniperus_seg_heterogeneous_20260905_111758"

VERDE, NARANJA, MORADO, AZUL = "#00E676", "#FF6D00", "#7B1FA2", "#2979FF"
COLS_EXP = ["#E69F00", "#0072B2", "#CC0044"]


def carga(base, exp, split, mec):
    f = f"{base}/{exp}/predictions/{split}/{mec}.json"
    return {im["image"]: im for im in json.load(open(f))["images"]}


def fondo(ax, split, nombre):
    """Las imagenes de campo son 422x338 y las de test 448x448. Se recorta una
    ventana cuadrada centrada para que todos los paneles salgan del mismo
    tamano."""
    im = np.asarray(Image.open(f"{IMG[split]}/{nombre.replace('.tif', '.jpg')}"))
    ax.imshow(im)
    h, w = im.shape[:2]
    lado = min(h, w)
    x0, y0 = (w - lado) / 2, (h - lado) / 2
    ax.set_xlim(x0, x0 + lado)
    ax.set_ylim(y0 + lado, y0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def gt(ax, polys):
    for g in polys:
        ax.add_patch(MplPoly(np.array(g).reshape(-1, 2), closed=True, fill=False,
                             edgecolor=VERDE, linewidth=0.9))


def pinta(ax, polys, scores, thr, color, lw=1.0):
    n = 0
    for p, s in zip(polys, scores):
        if s < thr:
            continue
        ax.add_patch(MplPoly(np.array(p).reshape(-1, 2), closed=True, fill=False,
                             edgecolor=color, linewidth=lw))
        n += 1
    return n


def caja(ax, polys, scores, thr, color, lw=1.0):
    n = 0
    for p, s in zip(polys, scores):
        if s < thr:
            continue
        a = np.array(p).reshape(-1, 2)
        x0, y0 = a.min(0); x1, y1 = a.max(0)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor=color, linewidth=lw))
        n += 1
    return n


def leyenda(fig, entradas, y=-0.055):
    fig.legend([Line2D([0], [0], color=c, lw=1.6) for _, c in entradas],
               [t for t, _ in entradas], loc="lower center",
               ncol=len(entradas), fontsize=6.5, bbox_to_anchor=(0.5, y),
               frameon=False)


# ----------------------------------------------------- 1. mascara frente a caja
def geometria():
    """Mask2Former: sus mascaras y las cajas que las envuelven, sobre la misma
    escena. Explica por que medirlo sobre cajas le cuesta 23 puntos en campo."""
    D = {}
    for sp, tmp in (("test", "test_tmp"), ("field_work", "fw_tmp")):
        pred = json.load(open("/home/santana/Documents/TFM/results/"
                              f"_batch_segmentation/{tmp}/swimt_processed_segm.json"))
        por = {}
        for p in pred:
            por.setdefault(int(p["image_id"]), ([], []))
            seg = p["segmentation"]
            poly = seg[0] if (isinstance(seg, list) and seg and isinstance(seg[0], list)) else seg
            por[int(p["image_id"])][0].append(poly)
            por[int(p["image_id"])][1].append(p["score"])
        D[sp] = por
    # el gt lo tomamos de las predicciones del bloque de tamano, que lo traen
    G = {sp: carga(SRV, TAM, sp, "dense_baseline") for sp in IMG}
    THR = {"test": 0.80, "field_work": 0.70}

    # escena con muchas detecciones vecinas, que es donde la caja estorba
    casos = []
    for sp in IMG:
        mejor, mv = None, -1
        for nom, d in G[sp].items():
            num = int(nom.split("_")[1].split(".")[0])
            if num not in D[sp]:
                continue
            po, sc = D[sp][num]
            n = sum(1 for s in sc if s >= THR[sp])
            ngt = len(d["gt"])
            # escena densa de vegetacion: muchas anotaciones y muchas detecciones
            v = min(n, ngt) if (8 <= ngt <= 22 and n >= 6) else -1
            if v > mv:
                mejor, mv = (nom, num), v
        casos.append((sp, mejor[0], mejor[1]))

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 1.02))
    for j, (sp, nom, num) in enumerate(casos):
        po, sc = D[sp][num]
        for i, modo in enumerate(("mascara", "caja")):
            ax = axes[i][j]
            fondo(ax, sp, nom); gt(ax, G[sp][nom]["gt"])
            if modo == "mascara":
                n = pinta(ax, po, sc, THR[sp], NARANJA)
            else:
                n = caja(ax, po, sc, THR[sp], MORADO)
            if i == 0:
                ax.set_title({"test": "Test (PI)", "field_work": "Campo (FW)"}[sp],
                             fontsize=8)
            if j == 0:
                ax.set_ylabel({"mascara": "Máscaras", "caja": "Cajas envolventes"}[modo],
                              fontsize=7)
    leyenda(fig, [("Anotación", VERDE), ("Predicción, máscara", NARANJA),
                  ("Predicción, caja envolvente", MORADO)])
    fig.suptitle("La misma predicción con las dos geometrías",
                 fontsize=9, fontweight="bold")
    guardar_png(fig, "ejemplo_geometria")



# --------------------------------------------------- 2. el umbral de confianza
def umbral():
    """El ensemble de expertos heterogeneo. Se marca en rojo lo que el umbral
    optimo retira y en naranja lo que conserva, de modo que se ve que lo que
    cae son detecciones sin arbusto debajo y no arbustos perdidos."""
    from shapely.geometry import Polygon as ShPoly
    from shapely.ops import unary_union
    OPT = {"test": 0.55, "field_work": 0.30}
    D = {sp: carga(EXP8, HET, sp, "btm_forest_ensemble_uniform") for sp in IMG}

    def sh(c):
        pts = list(zip(c[0::2], c[1::2]))
        if len(pts) < 3:
            return None
        g = ShPoly(pts)
        return g if g.is_valid else g.buffer(0)

    def analiza(d, opt):
        """(n_conserva, n_retira, n_retira_sin_arbusto, n_arbustos_perdidos)"""
        gts = [g for g in (sh(x) for x in d["gt"]) if g is not None and g.area > 0]
        cons, ret = [], []
        for po, sc in zip(d["polys"], d["scores"]):
            g = sh(po)
            if g is None or g.area == 0 or sc < 0.25:
                continue
            (cons if sc >= opt else ret).append(g)
        sin_arbusto = sum(1 for r in ret if not any(r.intersects(g) for g in gts))
        u = unary_union(cons) if cons else None
        perdidos = 0
        for g in gts:
            antes = any(g.intersects(r) for r in ret) or (u is not None and g.intersects(u))
            ahora = u is not None and g.intersects(u)
            if antes and not ahora:
                perdidos += 1
        return len(cons), len(ret), sin_arbusto, perdidos

    casos = []
    for sp in IMG:
        mejor, mv = None, -1
        for nom, d in D[sp].items():
            if not (10 <= len(d["gt"]) <= 26):
                continue
            nc, nr, ns, npd = analiza(d, OPT[sp])
            if nr < 3:
                continue
            v = ns - 3 * npd          # premia falsos positivos retirados, penaliza perdidas
            if v > mv:
                mejor, mv = (nom, nc, nr, ns, npd), v
        casos.append((sp, mejor))

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 0.56))
    for j_, (sp, (nom, nc, nr, ns, npd)) in enumerate(casos):
        ax = axes[j_]
        d = D[sp][nom]
        fondo(ax, sp, nom); gt(ax, d["gt"])
        for po, sc in zip(d["polys"], d["scores"]):
            if sc < 0.25:
                continue
            col = NARANJA if sc >= OPT[sp] else "#D50000"
            ax.add_patch(MplPoly(np.array(po).reshape(-1, 2), closed=True, fill=False,
                                 edgecolor=col, linewidth=1.0,
                                 linestyle="-" if sc >= OPT[sp] else (0, (2, 1.2))))
        ax.set_title({"test": "Test (PI)", "field_work": "Campo (FW)"}[sp], fontsize=8)
        ax.set_xlabel(f"{nc} conservadas, {nr} retiradas, "
                      f"{npd} arbustos perdidos", fontsize=6.5)
    leyenda(fig, [("Anotación", VERDE), ("Detección conservada", NARANJA),
                  ("Detección que el umbral retira", "#D50000")], y=-0.09)
    fig.suptitle("Qué detecciones retira el umbral óptimo", fontsize=9, fontweight="bold")
    guardar_png(fig, "ejemplo_umbral")


# ------------------------------------------------- 3. que ve cada experto
def expertos():
    """Los tres expertos por tamano sobre la misma escena. Cada uno detecta su
    rango y ninguno detecta el de los otros, que es el acuerdo nulo medido."""
    MECS = [("expert_E_small_solo", "Experto pequeño"),
            ("expert_E_medium_solo", "Experto mediano"),
            ("expert_E_large_solo", "Experto grande")]
    D = {sp: {m: carga(SRV, TAM, sp, m) for m, _ in MECS} for sp in IMG}
    casos = []
    for sp in IMG:
        mejor, mv = None, -1
        comun = set.intersection(*[set(D[sp][m]) for m, _ in MECS])
        for nom in comun:
            ns = [sum(1 for x in D[sp][m][nom]["scores"] if x >= 0.25) for m, _ in MECS]
            ngt = len(D[sp][MECS[0][0]][nom]["gt"])
            v = min(ns) if (8 <= ngt <= 22 and min(ns) >= 1) else -1
            if v > mv:
                mejor, mv = nom, v
        casos.append((sp, mejor))

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, TEXTWIDTH * 0.70))
    for i, (sp, nom) in enumerate(casos):
        for j, (mec, etq) in enumerate(MECS):
            ax = axes[i][j]
            d = D[sp][mec][nom]
            fondo(ax, sp, nom); gt(ax, d["gt"])
            n = pinta(ax, d["polys"], d["scores"], 0.25, COLS_EXP[j])
            ax.set_xlabel(f"{n} detecciones", fontsize=6.2)
            if i == 0:
                ax.set_title(etq, fontsize=7.5)
            if j == 0:
                ax.set_ylabel({"test": "Test (PI)", "field_work": "Campo (FW)"}[sp],
                              fontsize=7)
    leyenda(fig, [("Anotación", VERDE)] +
            [(e, COLS_EXP[k]) for k, (_, e) in enumerate(MECS)], y=-0.10)
    fig.suptitle("Qué detecta cada experto del reparto por tamaño",
                 fontsize=9, fontweight="bold")
    guardar_png(fig, "ejemplo_expertos")


# --------------------------------------- 4. el desplome del model merging
def merging():
    """Los mismos expertos combinados de las dos maneras. El ensemble los suma,
    el model merging los promedia y deja de detectar."""
    D = {sp: {m: carga(SRV, TAM, sp, m)
              for m in ("btm_forest_ensemble_uniform", "btm_forest_soup")} for sp in IMG}
    casos = []
    for sp in IMG:
        mejor, mv = None, -1
        for nom in D[sp]["btm_forest_ensemble_uniform"]:
            ne = sum(1 for x in D[sp]["btm_forest_ensemble_uniform"][nom]["scores"] if x >= 0.25)
            ns = sum(1 for x in D[sp]["btm_forest_soup"][nom]["scores"] if x >= 0.25)
            ngt = len(D[sp]["btm_forest_ensemble_uniform"][nom]["gt"])
            v = (ne - ns) if 8 <= ngt <= 22 else -1
            if v > mv:
                mejor, mv = nom, v
        casos.append((sp, mejor))

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH, TEXTWIDTH * 1.02))
    for j, (sp, nom) in enumerate(casos):
        for i, (mec, etq, col) in enumerate(
                (("btm_forest_ensemble_uniform", "Ensemble de expertos", NARANJA),
                 ("btm_forest_soup", "Model merging", MORADO))):
            ax = axes[i][j]
            d = D[sp][mec][nom]
            fondo(ax, sp, nom); gt(ax, d["gt"])
            n = pinta(ax, d["polys"], d["scores"], 0.25, col)
            ax.set_xlabel(f"{n} detecciones", fontsize=6.5)
            if i == 0:
                ax.set_title({"test": "Test (PI)", "field_work": "Campo (FW)"}[sp],
                             fontsize=8)
            if j == 0:
                ax.set_ylabel(etq, fontsize=7)
    leyenda(fig, [("Anotación", VERDE), ("Ensemble de expertos", NARANJA),
                  ("Model merging", MORADO)])
    fig.suptitle("Los mismos expertos, combinados de las dos maneras",
                 fontsize=9, fontweight="bold")
    guardar_png(fig, "ejemplo_merging")


if __name__ == "__main__":
    geometria(); umbral(); expertos(); merging()
