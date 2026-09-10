"""Reúne las conversiones auxiliares compartidas por los scripts de evaluación.

Biblioteca mínima que otros módulos importan: pasa una caja de formato xywh a xyxy, la
convierte en un polígono de shapely y extrae el número que identifica a cada imagen dentro
de su nombre de fichero.
"""

import re

from shapely.geometry import box

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group()) if match else -1


def box_to_poly(xyxy):
    x1, y1, x2, y2 = xyxy
    return box(x1, y1, x2, y2)