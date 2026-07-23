# Detección de objetos con MATLAB y despliegue en NVIDIA Jetson (ROS 2)

Suite en MATLAB R2025b para **entrenar (o exportar) un detector de
objetos tiny-YOLOv4**, **desplegarlo como nodo ROS 2 acelerado por GPU**
(código CUDA generado con GPU Coder e inferencia acelerada con cuDNN o
TensorRT) en una **Jetson Orin Nano Super Developer Kit (8 GB)** con
cámara USB, y **visualizar las detecciones en tiempo real desde MATLAB**.
Es la suite hermana del proyecto de Clasificación: misma estructura,
misma lógica de contrato y misma preparación de la Jetson.

## Arquitectura

```
        HOST (MATLAB R2025b)                 JETSON ORIN NANO (JetPack 7.2, ROS 2 Jazzy)
 ┌────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ trainDetector.m            │        │  cámara USB                              │
 │   └► trainedDetector.mat   │        │     └► v4l2_camera ──► /image_raw        │
 │ deployDetectorJetson.m ────┼──────► │  nodo detector_jetson (CUDA + cuDNN)     │
 │   (codegen + colcon remoto)│        │     ├► /detector/image_annotated         │
 │ viewDetectorROS2.m ◄───────┼─(DDS)──┤     └► /detector/detections              │
 └────────────────────────────┘        └──────────────────────────────────────────┘
```

La comunicación host ↔ Jetson es ROS 2 nativo (DDS): ambos equipos solo
necesitan compartir la misma red y el mismo `ROS_DOMAIN_ID`.

## Archivos (estructura plana)

```
trainDetector.m            # (1) entrena por transferencia o exporta tiny-YOLOv4/COCO
detectImage.m              # función de detección genérica en el host
testDetector.m             # (2) prueba local con una imagen estática
detectorJetsonNode.m       # (3) entry-point del nodo ROS 2 (codegen CUDA)
deployDetectorJetson.m     # (3) generación de código + despliegue en la Jetson
viewCameraROS2.m           # (4) verificación del flujo de cámara vía ROS 2
viewDetectorROS2.m         # (4) visualización: imagen anotada + etiquetas + FPS
loadYoloDataConfig.m       # helper: parser del data.yaml (Ultralytics)
readYoloLabels.m           # helper: TXT normalizado -> cajas [x y w h] en píxeles
buildYoloDatastore.m       # helper: dataset YOLO -> datastores de MATLAB
augmentYoloData.m          # helper: aumento de datos de entrenamiento
ensureRGBData.m            # helper: normalización a 3 canales
README_ObjectDetection.md  # este documento
README_ROS_OD.md           # configuración de la Jetson (ROS 2 + cámara USB)
```

## Nota sobre la arquitectura del detector

MATLAB no incluye YOLOv5 de forma nativa. Se emplea **tiny-YOLOv4**
(Computer Vision Toolbox) como equivalente ligero: tamaño y velocidad
comparables a YOLOv5n, entrenamiento nativo (`trainYOLOv4ObjectDetector`)
y soporte completo de generación de código CUDA, requisito del
despliegue. Al ser una red totalmente convolucional cabe con holgura en
los 8 GB compartidos de la Orin Nano.

## Modos de `trainDetector.m`

- **`mode = "export"`**: guarda tiny-YOLOv4 preentrenado tal cual (80
  clases de COCO: persona, botella, taza, laptop, celular...) en
  `trainedDetector.mat`. No requiere dataset. Ideal para demos y para
  validar el flujo completo de despliegue.
- **`mode = "train"`**: transferencia de aprendizaje sobre un dataset
  propio en formato YOLO/Ultralytics, con estimación automática de
  *anchor boxes* (k-means sobre IoU), aumento de datos y **evaluación
  integrada** (mAP@0.5 y mAP@[.5:.95] con `evaluateObjectDetection`,
  métricas por clase y mosaico cualitativo exportado).

Ambos modos guardan el **mismo contrato** en el MAT (`detector`,
`classNames`, `inputSize`, `baseModel`, `mode`, `info`, `testMetrics`),
por lo que el resto de la suite funciona igual con cualquiera de los dos.

## Formato del dataset (modo "train")

Formato YOLO/Ultralytics (el que exporta Roboflow):

```
datasetDir/
├── data.yaml                       # path, train, val, test, nc, names
├── train/images/*.jpg|png|bmp
├── train/labels/*.txt              # una caja por línea:
├── valid/images + valid/labels     #   id_clase cx cy w h  (normalizados 0-1)
└── test/images  + test/labels      # (opcional)
```

En `data.yaml`, si el export usa la carpeta `valid` (default de
Roboflow), declararlo: `val: valid`. Si falta `test`, las métricas se
calculan sobre `val` con un aviso. El bloque `roboflow:` extra del YAML
se ignora sin error. `buildYoloDatastore` imprime un resumen con las
cajas por clase: revisarlo para detectar desbalance antes de entrenar.

## Requisitos

- **Host**: MATLAB R2025b con Deep Learning Toolbox, Computer Vision
  Toolbox, MATLAB Coder, GPU Coder y ROS Toolbox. Add-ons: *Computer
  Vision Toolbox Model for YOLO v4 Object Detection*, *GPU Coder
  Interface for Deep Learning* y, opcional para la verificación previa,
  *MATLAB Coder Support Package for NVIDIA Jetson and NVIDIA DRIVE
  Platforms*.
- **Jetson**: JetPack 7.2, ROS 2 Jazzy, workspace colcon
  (`~/detector_ws`) y driver `v4l2_camera`. La preparación completa, con
  comandos listos para copiar y pegar, está en **`README_ROS_OD.md`**.
  Si la Jetson ya fue preparada para el proyecto de Clasificación, solo
  falta crear el workspace nuevo (sección 6 de esa guía).

## Orden de ejecución

### A. Preparación única (una sola vez por Jetson)

Seguir **`README_ROS_OD.md`** completo (secciones 1–10).

### B. Preparación del modelo (host; repetir solo si el modelo cambia)

1. `trainDetector.m` — elegir `mode` (`"export"` para tiny-YOLOv4/COCO
   sin dataset; `"train"` para transferencia con dataset propio, con
   métricas mAP y mosaico cualitativo). Produce `trainedDetector.mat`.
2. `testDetector.m` — validación local con una imagen estática.

### C. Despliegue (host; repetir solo si el modelo o el nodo cambian)

1. Verificar que `FRAME_SIZE` en `detectorJetsonNode.m` coincide con el
   `image_size` del driver de cámara ([480 640] para 640×480).
2. Editar IP/credenciales en `deployDetectorJetson.m` y ejecutarlo.
   Compila remotamente con colcon (~1–2 min) y **deja el nodo corriendo**.

### D. Puesta en marcha (cada sesión de trabajo)

1. **(Jetson)** Lanzar el driver de cámara — omitir si se configuró el
   servicio systemd de `README_ROS_OD.md`, sección 9:

   ```bash
   ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[640,480]" -p output_encoding:="rgb8"
   ```

2. **(Jetson)** Si el nodo no está corriendo (p. ej. tras un reinicio;
   inmediatamente después del despliegue ya está activo), relanzarlo.
   Verificar primero con `ros2 node list` que no exista ya (lanzar una
   segunda instancia duplica el nodo; ante duplicados,
   `pkill -f detectorJetsonNode` y lanzar uno solo):

   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/detector_ws/install/setup.bash
   ros2 run detectorjetsonnode detectorJetsonNode
   ```

   El nombre del paquete va en minúsculas pero el **ejecutable conserva
   las mayúsculas** del entry-point ("No executable found" es el síntoma
   de escribirlo todo en minúsculas). Ante la duda:
   `ros2 pkg executables detectorjetsonnode`.

3. **(Jetson)** Verificaciones:

   ```bash
   ros2 node list                          # debe aparecer /detector_jetson
   ros2 topic hz /image_raw                # ~20-30 Hz (cámara publicando)
   ros2 topic hz /detector/detections      # similar (el nodo SOLO publica si recibe imágenes)
   ```

4. **(Host, MATLAB)** Visualización en vivo — mismo `ROS_DOMAIN_ID` y
   misma red que la Jetson:

   ```matlab
   run("viewDetectorROS2.m")     % cajas + etiquetas + FPS; salir con 'q'
   ```

   Herramienta auxiliar: `viewCameraROS2.m` muestra solo `/image_raw`
   (útil para aislar problemas de red/driver de los del nodo).

## Tópicos ROS 2 publicados por el nodo

| Tópico | Tipo | Contenido |
|---|---|---|
| `/detector/image_annotated` | `sensor_msgs/Image` (`rgb8`) | Cuadro de cámara con las cajas dibujadas (colores por clase) |
| `/detector/detections` | `std_msgs/Float32MultiArray` | N×6 aplanado: `[x y w h score idClase]` por detección |

Las coordenadas `[x y w h]` están en píxeles de la imagen publicada
(`FRAME_SIZE`, 1-based) e `idClase` es el índice 1-based en `classNames`.
El texto de las etiquetas se superpone en el host (renderizar texto en
código embebido es costoso y restrictivo).

## Notas técnicas

- **Inferencia con cuDNN (FP32) por defecto**: es la ruta validada con
  JetPack 7.2. TensorRT falló en la suite de clasificación porque
  TensorRT 10.16 eliminó la API de la capa *fully-connected* que R2025b
  aún usa; tiny-YOLOv4 no tiene capas de ese tipo, así que
  `inferenceLibrary = "tensorrt"` es un experimento válido documentado
  en `deployDetectorJetson.m` (revertir a `"cudnn"` si falla).
- **Compute capability 8.7** declarada en el despliegue: el default de
  GPU Coder (sm_50) fue eliminado del nvcc de CUDA 13.
- **Tamaños deterministas para codegen**: el nodo fija canales con
  `I(:,:,1:3)` y dimensiones con `imresize` a `FRAME_SIZE` constante;
  `detect()` redimensiona internamente a la entrada de la red y devuelve
  las cajas en coordenadas del cuadro.
- **Red**: se recomienda Ethernet entre host y Jetson; el video `rgb8`
  sin comprimir puede saturar redes WiFi débiles.
