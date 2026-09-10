# 🌲 JuniperDetector

**Delineación individual de enebros y sabinas en la alta montaña de Sierra Nevada a partir de imágenes de satélite de muy alta resolución.**

Código del Trabajo Fin de Máster *Exploración de estrategias de optimización para la delineación individual de arbustos*, Máster Universitario en Ciencia de Datos e Ingeniería de Computadores, Universidad de Granada.

---

## 🎯 El problema

El enebro (*Juniperus communis*) y la sabina rastrera (*Juniperus sabina*) dominan la alta montaña de Sierra Nevada, forman parte de hábitats de interés prioritario para la conservación y se regeneran muy poco. Seguir su evolución exige censarlos **individuo a individuo** sobre superficies que los inventarios clásicos por transectos y parcelas no alcanzan.

Delinearlos de forma automática es difícil por dos motivos. Un mismo taxón adopta morfologías muy distintas según su edad y su emplazamiento, y anotarlos exige botánicos y campañas de campo, lo que limita el conjunto disponible a **570 imágenes**.

Este trabajo parte del modelo base de Khaldi et al. (2024) y explora cinco vías de optimización sobre sus mismos datos.

| | Vía explorada |
|---|---|
| 1 | Arquitecturas basadas en *transformers* (DETR, Co-DETR, Co-DINO, Mask2Former) |
| 2 | *Ensembles* de expertos especializados |
| 3 | *Model merging* sobre esos mismos expertos |
| 4 | Ampliación de la señal de entrenamiento sin datos nuevos |
| 5 | Protocolo de inferencia y calibración del umbral de confianza |

---

## 📈 Resultados

Todo se mide con el **F1 sobre S-IoU al 50 % de solapamiento**, sobre los dos conjuntos de test del trabajo de referencia: uno fotointerpretado, que comparte procedimiento de anotación con el entrenamiento, y otro de trabajo de campo, anotado sobre el terreno en emplazamientos alejados que no intervinieron en el desarrollo.

| | Test fotointerpretado | Test de trabajo de campo |
|---|---|---|
| Referencia publicada | 87,87 % | 76,86 % |
| **Mejor resultado de este trabajo** | **90,56 %** | **82,87 %** |
| Diferencia | **+2,69** | **+6,01** |

De los veintidós mecanismos evaluados, **catorce superan la referencia en cada uno de los dos conjuntos y trece la superan en ambos**, y lo hacen con modelos un orden de magnitud más pequeños que la arquitectura de referencia.

Los tres hallazgos que sostienen el trabajo:

- La especialización solo resulta eficaz cuando el criterio **reparte las anotaciones y no las imágenes**.
- *Ensemble* y *model merging* son **complementarios y no alternativos**: cada uno funciona en el régimen en el que el otro falla, y lo que decide cuál conviene es cuánto llegan a diferenciarse por dentro los modelos combinados, algo medible antes de combinarlos.
- El **umbral de confianza** mueve el resultado hasta treinta y tres puntos sobre unas mismas predicciones, más que la elección del propio mecanismo.

---

## 🗂️ Estructura

```
datos/          preparación del conjunto y control de calidad de las anotaciones
entrenamiento/  esquema Branch-Train-Merge, expertos y combinación
evaluacion/     métricas S-IoU, reevaluación y calibración del umbral
resultados/     métricas de todas las ejecuciones, en CSV
memoria/        generadores de las figuras y tablas del documento
```

---

## 🌱 datos/

Preparación del conjunto y control de calidad de las anotaciones.

| Script | Qué hace |
|---|---|
| `convert.py` | Convierte las anotaciones de partida en ficheros COCO con una anotación por arbusto. |
| `prepare_dataset.py` | Convierte el dataset COCO de detección al formato YOLO que espera Ultralytics. |
| `prepare_dataset_segmentation.py` | Convierte el dataset COCO de polígonos al formato YOLO-seg de segmentación de instancias. |
| `bucket_labels_by_size.py` | Genera tres subconjuntos YOLO por tamaño de caja a partir de un dataset ya convertido. |
| `verify_boxes.py` | Dibuja las cajas YOLO sobre sus imágenes para comprobar visualmente la conversión. |
| `verify_polygons.py` | Dibuja los polígonos YOLO-seg sobre sus imágenes para comprobar visualmente la conversión. |
| `check_cocofile.py` | Inspecciona un fichero de anotaciones COCO y dibuja las anotaciones de una imagen de ejemplo. |

---

## 🧠 entrenamiento/

| Script | Qué hace |
|---|---|
| `train_btm.py` | Ejecuta el pipeline Branch-Train-Merge de expertos YOLO para la detección de Juniperus. |

`train_btm.py` es el núcleo del trabajo. Implementa el esquema **Branch-Train-Merge**: entrena un modelo semilla común, deriva de él varios expertos que se entrenan por separado sobre distintas facetas del problema, y los combina de dos maneras opuestas. El *ensemble* fusiona las predicciones con **Weighted Box Fusion**, con un umbral de solapamiento de 0,55 entre cajas y ponderación uniforme o por fiabilidad. El *model merging* promedia directamente los parámetros y deja un único modelo, con el coste de inferencia de uno solo.

---

## 📐 evaluacion/

Métricas, reevaluación de las predicciones y calibración del umbral de confianza.

| Script | Qué hace |
|---|---|
| `siou_metrics_polygon.py` | Calcula las métricas IoU y S-IoU sobre geometría de polígono real mediante shapely. |
| `polygon_utils.py` | Reúne las utilidades geométricas de polígono usadas en el pipeline de segmentación. |
| `process_results.py` | Prepara las predicciones crudas del detector para la evaluación. |
| `analize_results.py` | Evalúa las detecciones de caja con IoU y S-IoU a lo largo del umbral de confianza. |
| `segmentation_analisis.py` | Evalúa las predicciones de segmentación barriendo el umbral de confianza con IoU y S-IoU. |
| `batch_evaluate_segmentation.py` | Evalúa en lote las máscaras de segmentación crudas de los dos conjuntos de test. |
| `reevaluar.py` | Reevalúa las predicciones crudas de cada experimento al umbral de confianza indicado. |
| `reevaluar_exp8.py` | Reevalúa las predicciones del Experimento 8 con la misma implementación de IoU y S-IoU. |
| `reevaluar_bbox.py` | Reevalúa las ejecuciones de detección convirtiendo sus cajas en polígonos de cuatro vértices. |
| `eval_legacy.py` | Evalúa con IoU y S-IoU las arquitecturas transformer a partir de sus detecciones crudas. |
| `eval_yolo_cajas.py` | Evalúa el modelo de segmentación con geometría de caja para compararlo con los detectores. |
| `eval_buckets.py` | Desglosa el rendimiento por tramo de tamaño real de los arbustos anotados. |
| `barrido_arriba.py` | Barre el umbral de confianza hacia arriba en las ejecuciones de segmentación. |
| `barrido_arriba_det.py` | Barre el umbral de confianza hacia arriba en las dos ejecuciones de detección. |
| `barrido_expertos.py` | Barre el umbral de confianza hacia arriba en los expertos por tamaño evaluados en solitario. |
| `barrido_expertos_region.py` | Barre el umbral de confianza hacia arriba en los expertos por región evaluados en solitario. |
| `barrido_confianza.py` | Barre el umbral de confianza completo en la única ejecución exportada a confianza muy baja. |
| `optimos.py` | Localiza el umbral de confianza óptimo de cada mecanismo a partir del barrido hacia arriba. |
| `calibrado.py` | Reúne los barridos de confianza y devuelve las cifras calibradas de cada mecanismo. |
| `unir_reevaluaciones.py` | Une la reevaluación general y la del Experimento 8 en un único fichero de métricas. |
| `extraer.py` | Consolida las métricas y la configuración de todos los experimentos del servidor. |
| `casos.py` | Clasifica cada instancia anotada según cómo la resolvió el modelo único. |
| `utils.py` | Reúne las conversiones auxiliares compartidas por los scripts de evaluación. |
| `visualize.py` | Dibuja sobre una imagen las anotaciones de referencia del conjunto seleccionado. |
| `visualize_results.py` | Dibuja sobre una imagen las predicciones del modelo junto a las anotaciones de referencia. |

Dos piezas merecen mención aparte. `siou_metrics_polygon.py` implementa la métrica **S-IoU** sobre geometría de polígono real, que a diferencia del IoU admite emparejamientos de muchos a muchos y por tanto no penaliza que dos fotointérpretes discrepen al delinear una colonia densa. `calibrado.py` es la fuente única de las cifras del trabajo: para cada ejecución, conjunto y mecanismo devuelve el mejor F1 y el umbral en el que se alcanza, de modo que ninguna comparación se hace a un umbral arbitrario.

---

## 📊 resultados/

Métricas de todas las ejecuciones en CSV, listas para reproducir cualquier tabla o figura del documento sin volver a entrenar ni a inferir.

| Fichero | Contenido |
|---|---|
| `metricas.csv` | Métricas consolidadas de todas las ejecuciones |
| `reeval_conf025.csv`, `reeval_conf025_completo.csv` | Reevaluación homogénea de todas las predicciones |
| `barrido_arriba.csv`, `barrido_arriba_det.csv` | Barrido del umbral de confianza, segmentación y detección |
| `barrido_expertos.csv`, `barrido_expertos_region.csv` | Barrido de los expertos por tamaño y por región |
| `optimos.csv` | Umbral óptimo y F1 de cada mecanismo |
| `legacy_modelos.csv`, `mask2former_mascaras.csv` | Arquitecturas *transformer* con las métricas de la referencia |
| `exp7_barrido.csv` | *Copy-paste* y destilación |
| `curvas_yolo.csv`, `curvas_legacy.csv` | Curvas de entrenamiento y validación |
| `participacion.csv` | Acuerdo entre expertos, arbusto a arbusto |
| `por_bucket.csv`, `casos.csv` | Desglose por tamaño y clasificación de los modos de fallo |
| `configuraciones.csv`, `yolo_cajas.csv` | Inventario de ejecuciones y evaluación sobre cajas |

---

## ⚙️ Puesta en marcha

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

El flujo completo, de las anotaciones al resultado:

```bash
python datos/convert.py                  # anotaciones de partida a COCO
python datos/prepare_dataset_segmentation.py   # COCO a formato YOLO-seg
python datos/bucket_labels_by_size.py    # subconjuntos de los expertos por tamaño
python entrenamiento/train_btm.py --stage all  # semilla, expertos, líneas base y combinación
python evaluacion/reevaluar.py           # métricas homogéneas de todas las predicciones
python evaluacion/barrido_arriba.py      # barrido del umbral de confianza
```

Los guiones esperan las rutas del entorno en el que se desarrolló el trabajo, así que hay que ajustar las constantes de cabecera antes de ejecutarlos. El conjunto de datos no se redistribuye aquí.

---

## 📄 memoria/

Guiones que generan las figuras y las tablas del documento a partir de los CSV de `resultados/`. No forman parte del método, solo de su presentación.

<details>
<summary>Ver el detalle</summary>

| Script | Qué hace |
|---|---|
| `_sel_polimorfismo.py` | Mide cada anotación del conjunto de entrenamiento para elegir recortes representativos. |
| `curvas_legacy.py` | Extrae las curvas de entrenamiento y validación de los experimentos de MMDetection. |
| `curvas_yolo.py` | Consolida en un único CSV las curvas de entrenamiento de todas las ejecuciones YOLO. |
| `estilo.py` | Define el estilo gráfico común a todas las figuras de la memoria. |
| `fig_atencion.py` | Dibuja el esquema de atención completa frente a atención deformable sobre una imagen real. |
| `fig_btm.py` | Genera las tres figuras del bloque de expertos: participación, curvas de entrenamiento y slicing. |
| `fig_calibracion_visual.py` | Dibuja las predicciones del model merging por tamaño a dos umbrales de confianza. |
| `fig_copypaste.py` | Reproduce el aumento por copy-paste sobre una escena real del conjunto de entrenamiento. |
| `fig_criterios.py` | Ilustra qué ve cada experto según el criterio de especialización. |
| `fig_cronograma.py` | Dibuja el cronograma del proyecto, de octubre de 2025 a septiembre de 2026. |
| `fig_curvas.py` | Dibuja las curvas de F1 frente al umbral de confianza de cada mecanismo. |
| `fig_dataset.py` | Muestra seis escenas del test fotointerpretado con las anotaciones superpuestas. |
| `fig_diagonal.py` | Dibuja el F1 de cada mecanismo por rango de tamaño del arbusto anotado. |
| `fig_ejemplos.py` | Genera las cuatro figuras de ejemplos reales que acompañan la discusión de resultados. |
| `fig_ensemble_soup.py` | Dibuja el esquema que contrapone el ensemble y el model soup. |
| `fig_escalas.py` | Muestra la misma escena a tres resoluciones con sus anotaciones superpuestas. |
| `fig_exp7.py` | Dibuja el barrido del umbral de confianza de los dos mecanismos que generan más señal. |
| `fig_expertos_tamano.py` | Dibuja la misma escena anotada tal como la ve cada uno de los tres expertos de tamaño. |
| `fig_global.py` | Sitúa cada mecanismo calibrado en el plano de F1 de los dos conjuntos de evaluación. |
| `fig_metricas.py` | Compara IoU y S-IoU sobre los tres casos que separan a las dos métricas. |
| `fig_polimorfismo.py` | Reúne recortes de anotaciones reales que ilustran los seis ejes de variabilidad del enebro. |
| `fig_predicciones.py` | Contrapone predicción y anotación sobre seis escenas de acierto y de fallo. |
| `fig_regimenes.py` | Genera la misma imagen bajo los tres regímenes de data augmentation que definen los expertos. |
| `fig_regiones.py` | Dibuja la distribución geográfica de las imágenes y su reparto en las dos regiones. |
| `fig_sahi.py` | Muestra cómo cambia la escala aparente de un arbusto pequeño al trocear la imagen. |
| `fig_senal.py` | Dibuja el esquema de las dos vías empleadas para generar más señal de entrenamiento. |
| `fig_workflow.py` | Dibuja el esquema del flujo de trabajo seguido en el proyecto. |
| `single_training_plot.py` | Dibuja la curva de métricas de validación de una sola ejecución a partir de su log. |
| `tablas.py` | Genera los fragmentos LaTeX de todas las tablas de la memoria a partir de los CSV de análisis. |

</details>

---

## 📚 Referencia

El trabajo parte de:

> Khaldi, R., Tabik, S., Puertas-Ruiz, S., Hernández-Lambraño, R. E., Rodríguez, J. A. D., Alcaraz-Segura, D., Herrera-Maldonado, F. (2024). *Individual mapping of large polymorphic shrubs in high mountains using satellite images and deep learning*. **ISPRS Journal of Photogrammetry and Remote Sensing**.

---

**Álvaro Santana Sánchez** · Tutora: Siham Tabik Ouled Hrour · Universidad de Granada, septiembre de 2026
