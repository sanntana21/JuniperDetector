"""Convierte las anotaciones de partida en ficheros COCO con una anotación por arbusto.

Según el modo elegido, reescribe el JSON de Train, Val o Test separando cada polígono de
segmentación en una anotación independiente con su caja, o construye el JSON del test de trabajo
de campo a partir de los shapefiles y los rásteres, reproyectando las geometrías a coordenadas de
píxel de cada imagen. El resultado se escribe como <modo>_updated.json en la carpeta Annotations
correspondiente y es la entrada de prepare_dataset.py.
"""

import os
import json
import argparse
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from datetime import datetime
import numpy as np

import geopandas as gpd
import rasterio
from shapely.ops import transform as shp_transform



def calculate_bbox(xs, ys):
    min_x, min_y = min(xs), min(ys)
    width = max(xs) - min_x
    height = max(ys) - min_y
    return [min_x, min_y, width, height]


def convert_segmentations_to_annotations(ann, base_id):
    annotations = []
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    iscrowd = ann["iscrowd"]

    for seg in ann["segmentation"]:
        xs = seg[::2]
        ys = seg[1::2]
        bbox = calculate_bbox(xs, ys)
        area = bbox[2] * bbox[3]

        annotations.append(
            {
                "id": base_id,
                "image_id": image_id,
                "category_id": category_id,
                "iscrowd": iscrowd,
                "area": area,
                "bbox": bbox,
                "segmentation": [seg],
                "width": max(xs) - min(xs),
                "height": max(ys) - min(ys),
            }
        )
        base_id += 1

    return annotations, base_id


# ---------- Helpers ----------
def normalize_to_image(geom, bounds, img_w=442, img_h=338):
    """Normalize geometry from shapefile bounds into fixed image size."""
    minx, miny, maxx, maxy = bounds
    scale_x = img_w / (maxx - minx)
    scale_y = img_h / (maxy - miny)

    def _map_to_pixel(x, y):
        px = (x - minx) * scale_x
        py = (maxy - y) * scale_y  # invert Y so origin at top-left
        return px, py

    return shp_transform(lambda x, y: _map_to_pixel(x, y), geom)

def polygon_segmentation_pixels_old(poly):
    """Convert polygon geometry into COCO segmentation (in pixel coords)."""
    segs = []

    def _ring_to_list(ring):
        coords = np.asarray(ring.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        return coords.reshape(-1).round(1).tolist()

    if poly.geom_type == "Polygon":
        segs.append(_ring_to_list(poly.exterior))
        for interior in poly.interiors:
            hole = _ring_to_list(interior)
            if len(hole) >= 6:
                segs.append(hole)
    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            segs.extend(polygon_segmentation_pixels_old(p))
    return segs

def polygon_segmentation_pixels(poly):
    segs = []

    def ring_to_list(ring):
        coords = np.asarray(ring.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if coords.shape[0] < 3:
            return None
        return coords.reshape(-1).round(1).tolist()

    if poly.geom_type == "Polygon":
        ext = ring_to_list(poly.exterior)
        if ext:
            segs.append(ext)
        for interior in poly.interiors:
            hole = ring_to_list(interior)
            if hole:
                segs.append(hole)

    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            segs += polygon_segmentation_pixels(p)

    return segs

def bbox_from_geom_pixels(geom, img_w=442, img_h=338, min_size=1.0):
    """Compute clipped bbox in pixel space [x, y, w, h]."""
    minx, miny, maxx, maxy = geom.bounds
    minx = max(0.0, min(minx, img_w - 1))
    miny = max(0.0, min(miny, img_h - 1))
    maxx = max(0.0, min(maxx, img_w - 1))
    maxy = max(0.0, min(maxy, img_h - 1))

    w = maxx - minx
    h = maxy - miny
    if w < min_size: w = min_size
    if h < min_size: h = min_size
    return [round(minx, 1), round(miny, 1), round(w, 1), round(h, 1)]

def shapefiles_to_coco_json_old(directory: str,  img_w=422, img_h=338, min_size=1.0, img_extension : str = "tif"):
    dataset = {
        "info": {
            "description": "Detector Dataset",
            "version": "0.2.0",
            "year": 2023,
            "contributor": "Generated",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "Detector License", "url": ""}],
        "categories": [{"id": 1, "name": "Juniperus", "supercategory": "Shrub"}],
        "images": [],
        "annotations": [],
    }

    # image_id = 0
    ann_id = 0
    for file in os.listdir(directory):
        if file.endswith(".shp"):
            file_id = file.split("_")[1].split(".")[0]
            gdf = gpd.read_file(os.path.join(directory, file))
            
            # Compute global bounds for normalization
            bounds = gdf.total_bounds
            with rasterio.open(image_path) as src:
                bounds = src.bounds
                img_w = src.width
                img_h = src.height
 
            
            image_id = int(file_id)
            # Example: assume each shapefile corresponds to one image
            dataset["images"].append(
                {
                    "id": image_id,
                    "file_name": f"Img_{file_id}.{img_extension}",
                    "width": img_w,  # TODO: replace with real values if available
                    "height": img_h,
                    "date_captured": datetime.now().isoformat(),
                    "license": 1,
                }
            )
            for _,row in gdf.iterrows():
                geom_px = normalize_to_image(row.geometry, bounds, img_w, img_h)
                bbox = bbox_from_geom_pixels(geom_px, img_w, img_h)
                segmentation = polygon_segmentation_pixels(geom_px)
                area = float(round(geom_px.area, 1))

                dataset["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "width": bbox[2],
                    "height": bbox[3]
                })
                
                ann_id += 1
                
            

    return dataset


def shapefiles_to_coco_json(
    shapefiles_dir: str,
    images_dir: str,
    img_extension: str = "tif"
):
    dataset = {
        "info": {
            "description": "Detector Dataset",
            "version": "0.3.0",
            "year": 2024,
            "contributor": "Generated",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [{"id": 1, "name": "Detector License", "url": ""}],
        "categories": [
            {"id": 1, "name": "Juniperus", "supercategory": "Shrub"}
        ],
        "images": [],
        "annotations": [],
    }

    image_id = 0
    ann_id = 0

    for shp_file in sorted(os.listdir(shapefiles_dir)):
        if not shp_file.endswith(".shp"):
            continue

        file_id = os.path.splitext(shp_file)[0].split("_")[-1]
        image_name = f"Img_{file_id}.{img_extension}"
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Imagen no encontrada: {image_name}")
            continue

        # --- abrir raster ---
        with rasterio.open(image_path) as src:
            img_w = src.width
            img_h = src.height
            raster_bounds = src.bounds
            raster_crs = src.crs

        # --- leer shapefile ---
        gdf = gpd.read_file(os.path.join(shapefiles_dir, shp_file))

        # --- reproyectar si es necesario ---
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        dataset["images"].append({
            "id": file_id,
            "file_name": image_name,
            "width": img_w,
            "height": img_h,
            "date_captured": datetime.now().isoformat(),
            "license": 1,
        })

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            geom_px = normalize_to_image(
                geom, raster_bounds, img_w, img_h
            )

            bbox = bbox_from_geom_pixels(geom_px, img_w, img_h)
            if bbox is None:
                continue

            segmentation = polygon_segmentation_pixels(geom_px)
            if not segmentation:
                continue

            area = float(abs(geom_px.area))

            dataset["annotations"].append({
                "id": ann_id,
                "image_id": file_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": round(area, 1),
                "bbox": bbox,
                "segmentation": segmentation,
            })

            ann_id += 1

        image_id += 1

    return dataset


mode_map = {0: "Train", 1: "Val", 2: "Test", 3: "FieldWork"}


def main():
    extension = ""
    parser = argparse.ArgumentParser(description="Selecciona el modo de ejecución.")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Modo: 0 = Train (por defecto), 1 = Val, 2 = Test, 3 = FieldWork (Shapefiles)",
    )
    args = parser.parse_args()

    mode_str = mode_map[args.mode]
    print(f"Modo seleccionado: {mode_str}")

    # Lógica según modo
    if mode_str == "Train":
        print("Ejecutando entrenamiento...")
    elif mode_str == "Val":
        print("Ejecutando validación...")
    elif mode_str == "Test":
        print("Ejecutando prueba...")
    elif mode_str == "FieldWork":
        print("Ejecutando transformacion de Shapefiles (FieldWork)")
    else:
        raise ValueError("El modo seleccionado no es valido.")
    
    new_annotations = []
    
    # Example usage:
    if mode_str == "FieldWork":

        annotations_dir = f"./Field_Work_Data{extension}/External_Val_Data/Annotations"
        images_dir = f"./Field_Work_Data{extension}/External_Val_Data/Images"
        
        shapefiles_dir = os.path.join(annotations_dir,"Shapefiles")
        dataset = shapefiles_to_coco_json(shapefiles_dir,images_dir=images_dir)
        print(dataset)

        out_path = os.path.join(annotations_dir, f"{mode_str}_updated.json")

        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
    else:
        # --- Settings ---
        annotations_dir = f"./Photo_Interpretation_Data{extension}/{mode_str}/Annotations"

        coco_path = os.path.join(annotations_dir, f"{mode_str}.json")
        out_path = os.path.join(annotations_dir, f"{mode_str}_updated.json")

        # --- Load existing COCO JSON ---
        with open(coco_path) as f:
            coco = json.load(f)

        # --- Initialize ---
        
        coco["images"] = [{k:(v if k != "file_name" else v.replace("tif","jpg")) for k,v in img.items()} for img in coco["images"]]
        ann_id = 0
        anns = [ann for ann in coco["annotations"]]

        for ann in anns:
            trans_ann, ann_id = convert_segmentations_to_annotations(ann, ann_id)
            new_annotations.extend(trans_ann)

        coco["annotations"] = new_annotations

        out_path = os.path.join(annotations_dir, f"{mode_str}_updated.json")
        with open(out_path, "w") as f:
            json.dump(coco, f, indent=4)

    print(f"✅ Saved updated COCO annotations to: {out_path}")
    print(f"➕ Added {len(new_annotations)} new grouped annotations.")


if __name__ == "__main__":
    main()


