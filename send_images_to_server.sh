#!/bin/bash
# Script: subir_coco_filtrado.sh
# Uso: ./subir_coco_filtrado.sh /ruta/local/Photo_Interpretation_Data

USER="asantana"
SERVER="ngpu.ugr.es"
DEST="/mnt/homeGPU/$USER/Co-DETR/data/juniper"

LOCAL_ROOT="$1"

if [ -z "$LOCAL_ROOT" ]; then
    echo "Error: Indica la ruta local del dataset."
    exit 1
fi

# Crear estructura remota
ssh ${USER}@${SERVER} "mkdir -p $DEST/annotations $DEST/train2017 $DEST/val2017"

# Subir anotaciones filtradas
echo "Subiendo anotaciones actualizadas..."
scp "$LOCAL_ROOT/Train/Annotations/"*"_updated.json" ${USER}@${SERVER}:$DEST/annotations/instances_train2017.json
scp "$LOCAL_ROOT/Val/Annotations/"*"_updated.json" ${USER}@${SERVER}:$DEST/annotations/instances_val2017.json

# Subir imágenes
echo "Subiendo imágenes de entrenamiento..."
scp -r "$LOCAL_ROOT/Train/Images/"* ${USER}@${SERVER}:$DEST/train2017/

echo "Subiendo imágenes de validación..."
scp -r "$LOCAL_ROOT/Val/Images/"* ${USER}@${SERVER}:$DEST/val2017/

echo "Transferencia completada."

