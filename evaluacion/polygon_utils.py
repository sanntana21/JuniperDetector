"""Reúne las utilidades geométricas de polígono usadas en el pipeline de segmentación.

Biblioteca sin dependencias externas que otros módulos importan: área por la fórmula del
shoelace, conversión entre píxeles y coordenadas normalizadas, caja envolvente y selección
de la parte de mayor área cuando una instancia COCO llega partida en varios polígonos.
"""


def polygon_area(points_flat):
    """Area de un poligono simple (formula del shoelace / Gauss).
    points_flat: lista plana [x1,y1,x2,y2,...,xn,yn] en cualquier unidad
    consistente (pixeles o normalizado -- el area sale en esa misma unidad
    al cuadrado)."""
    n = len(points_flat) // 2
    if n < 3:
        return 0.0
    xs = points_flat[0::2]
    ys = points_flat[1::2]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def normalize_polygon(points_flat, width, height):
    """[x1,y1,...] en pixeles absolutos -> normalizado 0-1."""
    out = []
    for i in range(0, len(points_flat), 2):
        out.append(points_flat[i] / width)
        out.append(points_flat[i + 1] / height)
    return out


def denormalize_polygon(points_flat, width, height):
    """[x1,y1,...] normalizado 0-1 -> pixeles absolutos."""
    out = []
    for i in range(0, len(points_flat), 2):
        out.append(points_flat[i] * width)
        out.append(points_flat[i + 1] * height)
    return out


def polygon_bbox(points_flat):
    """Bounding box [x1,y1,x2,y2] (misma unidad que la entrada) que
    envuelve al poligono -- util para filtros/visualizacion rapida."""
    xs = points_flat[0::2]
    ys = points_flat[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def largest_polygon_part(segmentation_parts):
    """Un objeto COCO puede tener varias partes de poligono disjuntas
    (ej. un shrub partido por oclusion). YOLO-seg solo admite UN poligono
    por instancia -- nos quedamos con la parte de mayor area, y devolvemos
    tambien cuantas partes habia (para poder avisar cuando se descarta
    informacion)."""
    if len(segmentation_parts) == 1:
        return segmentation_parts[0], 1
    areas = [polygon_area(p) for p in segmentation_parts]
    best_idx = max(range(len(areas)), key=lambda i: areas[i])
    return segmentation_parts[best_idx], len(segmentation_parts)
