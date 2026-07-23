# Clasificación de imágenes con MATLAB y despliegue en NVIDIA Jetson (ROS 2)

Suite en MATLAB R2025b para **entrenar (o exportar) un clasificador de
imágenes**, **desplegarlo como nodo ROS 2 acelerado por GPU** (código CUDA
generado con GPU Coder e inferencia acelerada con cuDNN o TensorRT) en una **Jetson Orin Nano
Super Developer Kit (8 GB)** con cámara USB, y **visualizar los resultados
en tiempo real desde MATLAB**.

## Arquitectura

```
        HOST (MATLAB R2025b)                 JETSON ORIN NANO (JetPack 7.2, ROS 2 Jazzy)  
 ┌────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ trainClassifier.m          │        │  cámara USB                              │
 │   └► trainedClassifier.mat │        │     └► v4l2_camera ──► /image_raw        │
 │ deployClassifierJetson.m ──┼──────► │  nodo classifier_jetson (CUDA + cuDNN)    │
 │   (codegen + colcon remoto)│        │     ├► /classifier/class_id              │
 │ viewClassifierROS2.m ◄─────┼─(DDS)──┤     ├► /classifier/scores                │
 └────────────────────────────┘        │     └► /classifier/image_view            │
                                       └──────────────────────────────────────────┘
```

La comunicación host ↔ Jetson es ROS 2 nativo (DDS): ambos equipos solo
necesitan compartir la misma red y el mismo `ROS_DOMAIN_ID`.

## Archivos (estructura plana)

```
trainClassifier.m          # (1) entrena por transferencia o exporta la red base
classifyImage.m            # función de inferencia genérica en el host
testClassifier.m           # (2) prueba local con una imagen estática
classifierJetsonNode.m     # (3) entry-point del nodo ROS 2 (codegen CUDA)
deployClassifierJetson.m   # (3) generación de código + despliegue en la Jetson
viewCameraROS2.m           # (4) verificación del flujo de cámara vía ROS 2
viewClassifierROS2.m       # (4) visualización: imagen + clase + top-3
README_clasificacion.md    # este documento
README_jetson_ros2.md      # configuración de la Jetson (ROS 2 + cámara USB)
```

## Modos de `trainClassifier.m`

- **`mode = "export"`**: guarda la red base preentrenada por defecto (1000
  clases de ImageNet) en `trainedClassifier.mat`. No requiere dataset, por 
  lo que es útil para validar el flujo completo de despliegue.
- **`mode = "train"`**: transferencia de aprendizaje sobre un dataset
  propio con particiones predefinidas.

Ambos modos guardan el **mismo contrato** en el MAT (`net`, `classNames`,
`inputSize`, `baseModel`, `mode`, `info`, `testAccuracy`), por lo que el
resto de la suite funciona igual con cualquiera de los dos.

## Formato del dataset (modo "train")

Particiones predefinidas; las etiquetas se toman del nombre de la
subcarpeta de clase:

```
datasetDir/
├── train/                  (obligatoria)
│   ├── clase_1/*.jpg|png|bmp
│   └── clase_2/...
├── valid/                  (o "val"; opcional)
│   └── ...
└── test/                   (opcional)
    └── ...
```

Si falta `valid`/`val`, el script separa automáticamente un 15 % de
`train` como validación. Si falta `test`, las métricas finales se calculan
sobre la validación (con un aviso: interpretarlas con cautela). El script
también verifica la coherencia de clases entre particiones y advierte si
alguna partición tiene clases desconocidas o faltantes.

## Requisitos

- **Host**: MATLAB R2025b con Deep Learning Toolbox, Computer Vision
  Toolbox, MATLAB Coder, GPU Coder y ROS Toolbox. Add-ons: el modelo base
  elegido (p. ej. *Deep Learning Toolbox Model for ResNet-18 Network*;
  `squeezenet` no requiere add-on), *GPU Coder Interface for Deep
  Learning* y, opcional para la verificación previa, *MATLAB Coder
  Support Package for NVIDIA Jetson and NVIDIA DRIVE Platforms*.
- **Jetson**: JetPack 7.2 (Ubuntu 24.04), ROS 2 Jazzy, workspace colcon y driver de
  cámara `v4l2_camera`. La preparación completa, con comandos listos para
  copiar y pegar, está en **`README_jetson_ros2.md`**.

## Orden de ejecución

### A. Preparación única (una sola vez por Jetson)

Seguir **`README_jetson_ros2.md`** completo (secciones 1–10): JetPack 7.2,
componentes del SDK, variables de entorno, ROS 2 Jazzy, workspace,
dominio y driver de cámara.

### B. Preparación del modelo (host; repetir solo si el modelo cambia)

1. `trainClassifier.m` — elegir `mode` (`"export"` para la red base con
   clases ImageNet sin dataset; `"train"` para transferencia con dataset
   propio). Produce `trainedClassifier.mat` y, en modo entrenamiento, la
   matriz de confusión (`confusion_test.png`).
2. `testClassifier.m` — validación local con una imagen estática (top-3).

### C. Despliegue (host; repetir solo si el modelo o el nodo cambian)

1. Verificar `INPUT_SIZE` en `classifierJetsonNode.m` (224×224×3 para
   ResNet-18; el script de despliegue lo comprueba automáticamente).
2. Editar IP/credenciales en `deployClassifierJetson.m` y ejecutarlo.
   Compila remotamente con colcon (~1–2 min) y **deja el nodo corriendo**
   en la Jetson.

### D. Puesta en marcha (cada sesión de trabajo)

1. **(Jetson)** Lanzar el driver de cámara — omitir si se configuró el
   servicio systemd de `README_jetson_ros2.md`, sección 9:

   ```bash
   ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[640,480]" -p output_encoding:="rgb8"
   ```

2. **(Jetson)** Si el nodo no está corriendo (p. ej. tras un reinicio;
   inmediatamente después del despliegue ya está activo), relanzarlo.
   Verificar primero con `ros2 node list` que no exista ya: lanzar una
   segunda instancia duplica el nodo (aviso de nombre repetido y tópicos
   al doble de frecuencia); ante duplicados, `pkill -f classifierJetsonNode`
   y lanzar uno solo:

   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/classifier_ws/install/setup.bash
   ros2 run classifierjetsonnode classifierJetsonNode
   ```

   El nombre del paquete va en minúsculas pero el **ejecutable conserva
   las mayúsculas** del entry-point ("No executable found" es el síntoma
   de escribirlo todo en minúsculas). Ante la duda, listar los
   ejecutables del paquete: `ros2 pkg executables classifierjetsonnode`.

3. **(Jetson)** Verificaciones:

   ```bash
   ros2 node list                        # debe aparecer /classifier_jetson
   ros2 topic hz /image_raw              # ~30 Hz (cámara publicando)
   ros2 topic hz /classifier/class_id    # ~30 Hz (el nodo SOLO publica si recibe imágenes)
   ```

4. **(Host, MATLAB)** Visualización en vivo — mismo `ROS_DOMAIN_ID` y
   misma red que la Jetson:

   ```matlab
   run("viewClassifierROS2.m")     % imagen + clase + top-3; salir con 'q'
   ```

   Herramienta auxiliar: `viewCameraROS2.m` muestra solo `/image_raw`
   (útil para aislar problemas de red/driver de los del nodo).

## Tópicos ROS 2 publicados por el nodo

| Tópico | Tipo | Contenido |
|---|---|---|
| `/classifier/class_id` | `std_msgs/UInt32` | Índice 1-based de la clase ganadora |
| `/classifier/scores` | `std_msgs/Float32MultiArray` | Probabilidades de todas las clases |
| `/classifier/image_view` | `sensor_msgs/Image` (`rgb8`) | Imagen de entrada a la red (depuración) |

## Notas técnicas

- **Rango de píxeles**: la red se alimenta con `single(im)` en rango
  0–255, sin `im2single`: la normalización (zerocenter/zscore) está
  integrada en la capa de entrada del `dlnetwork` y sus estadísticas se
  calcularon en esa escala.
- **Preprocesamiento unificado**: entrenamiento, prueba local y nodo de
  la Jetson aplican el mismo redimensionado directo a la entrada de la
  red; esta coherencia es la que hace que la precisión desplegada
  coincida con la medida en test.
- **`INPUT_SIZE` fijo**: la generación de código exige tamaño de entrada
  constante en compilación (requisito del despliegue GPU, tanto con
  cuDNN como con TensorRT), por eso es una constante del entry-point
  (verificada automáticamente por `deployClassifierJetson.m` contra la
  red).
- **Red**: se recomienda Ethernet entre host y Jetson; el video `rgb8`
  sin comprimir puede saturar redes WiFi débiles.
