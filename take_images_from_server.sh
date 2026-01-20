#!/bin/bash
# Script take_images_from_server.sh
# Uso: ./stake_images_from_server /ruta/destino /ruta/local

USER="asantana"
SERVER="ngpu.ugr.es"
DEST_ROOT="$1"

if [ -z "$DEST_ROOT" ]; then
    echo "Error: Indica la ruta remota que copiar"
    exit 1
fi

LOCAL_ROOT="$2"

if [ -z "$LOCAL_ROOT" ]; then
    echo "Error: Indica la ruta donde copiar los archivos"
    exit 1
fi

# Normaliza ruta remota (añade / si falta)
if [[ "$DEST_ROOT" != */ ]]; then
    DEST_ROOT="${DEST_ROOT}/"
fi


echo "Conectándose al servidor ${SERVER} como ${USER}..."
ssh "${USER}@${SERVER}" "echo 'Conexión establecida con éxito.'"

# Copia de la carpeta deseada
echo "Iniciando transferencia desde ${DEST_ROOT} a ${LOCAL_ROOT}/..."
scp -r "${USER}@${SERVER}:${DEST_ROOT}" "${LOCAL_ROOT}/"


if [ $? -eq 0 ]; then
    echo "Transferencia completada con éxito."
else
    echo "Error durante la transferencia."
fi
