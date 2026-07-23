# Despliegue de modelos de Deep Learning en NVIDIA Jetson Orin Nano con MATLAB y ROS 2

Este repositorio documenta y distribuye los recursos necesarios para implementar **clasificadores de imágenes** y **detectores de objetos** en una *NVIDIA Jetson Orin Nano Super Developer Kit (8 GB)* mediante **MATLAB R2025b**, **GPU Coder**, **CUDA 13**, **cuDNN**/**TensorRT** y **ROS 2 Jazzy**. Incluye scripts de MATLAB para entrenar y exportar redes, la generación automática de **nodos ROS 2 completos** desde MATLAB (sin escribir C++), y visualizadores para observar la inferencia en tiempo real desde el PC host.

Todo el flujo fue **verificado en la práctica** sobre JetPack 7.2 (Ubuntu 24.04, CUDA 13.2, cuDNN 9.20, TensorRT 10.16), una pila más reciente que la validada oficialmente por MathWorks; los ajustes necesarios están incorporados en los scripts y documentados en las guías.

## Descripción general del proyecto

El objetivo es demostrar cómo llevar modelos de deep learning desarrollados en MATLAB a un dispositivo embebido Jetson y operarlos como **nodos ROS 2 nativos** en tiempo real. Se abordan tres escenarios:

1. **Clasificación**: transferencia de aprendizaje con `imagePretrainedNetwork` + `trainnet` (ResNet-18, MobileNetV2, SqueezeNet...) sobre un dataset propio, o exportación directa de la red base con sus 1000 clases de ImageNet. Incluye además un ejercicio didáctico completo con **DigitDataset/MNIST** (`trainMNIST.m`): CNN sencilla (conv + tanh + densa + softmax) con análisis del modelo y variante conv + ReLU + maxpool + dropout.
2. **Detección de objetos**: **tiny-YOLOv4** (equivalente ligero a YOLOv5n con soporte nativo de codegen), en modo exportación (80 clases de COCO) o entrenamiento por transferencia sobre datasets en formato **YOLO/Ultralytics** (data.yaml + TXT, el que exporta Roboflow), con estimación de *anchor boxes*, aumento de datos y evaluación mAP integrada.
3. **Despliegue y visualización**: GPU Coder + ROS Toolbox generan el nodo ROS 2 completo, lo transfieren a la Jetson y lo compilan allí con colcon (compilación remota: no se necesita CUDA en el host). La comunicación host ↔ Jetson es DDS puro (sin maestro ROS): basta compartir red y `ROS_DOMAIN_ID`.

```
        HOST (MATLAB R2025b)                 JETSON ORIN NANO (JetPack 7.2, ROS 2 Jazzy)
 ┌────────────────────────────┐        ┌──────────────────────────────────────────┐
 │ entrenamiento / export     │        │  cámara USB ─► v4l2_camera ─► /image_raw │
 │   └► modelo .mat (contrato)│        │  nodo generado (CUDA + cuDNN)            │
 │ script de despliegue ──────┼──────► │     ├► tópicos de resultados             │
 │ visualizador ROS 2 ◄───────┼─(DDS)──┤     └► imagen anotada                    │
 └────────────────────────────┘        └──────────────────────────────────────────┘
```

Cada suite trabaja por **contrato**: el script de entrenamiento guarda un `.mat` con variables estandarizadas (`net`/`detector`, `classNames`, `inputSize`, ...) que el resto de los archivos consume, de modo que cambiar de modelo no requiere tocar el despliegue ni los visualizadores.

## Estructura del repositorio

### `Classification/`

Clasificación de imágenes: entrenamiento/exportación, prueba local, nodo ROS 2 y visualización.

```
Classification/
├── trainClassifier.m          # Entrena por transferencia o exporta la red base (ImageNet)
├── trainMNIST.m               # Ejercicio MNIST/DigitDataset con análisis del modelo
├── adaptStudentModel.m        # Adapta un MAT con solo 'net' al contrato del pipeline
├── classifyImage.m            # Función de inferencia genérica en el host
├── testClassifier.m           # Prueba local con una imagen estática (top-3)
├── classifierJetsonNode.m     # Entry-point del nodo ROS 2 (codegen CUDA)
├── deployClassifierJetson.m   # Generación de código + despliegue en la Jetson
├── viewCameraROS2.m           # Verificación del flujo de cámara vía ROS 2
├── viewClassifierROS2.m       # Visualización: imagen + clase + top-3
├── README_clasificacion.md    # Documentación de la suite
└── README_jetson_ros2.md      # Preparación de la Jetson (comandos copy-paste)
```

Tópicos publicados: `/classifier/class_id` (UInt32), `/classifier/scores` (Float32MultiArray) y `/classifier/image_view` (imagen de entrada a la red).

### `ObjectDetection/`

Detección de objetos con tiny-YOLOv4: mismo esquema y misma lógica de contrato.

```
ObjectDetection/
├── trainDetector.m            # Entrena por transferencia o exporta tiny-YOLOv4 (COCO)
├── detectImage.m              # Función de detección genérica en el host
├── testDetector.m             # Prueba local con una imagen estática
├── detectorJetsonNode.m       # Entry-point del nodo ROS 2 (codegen CUDA)
├── deployDetectorJetson.m     # Generación de código + despliegue en la Jetson
├── viewCameraROS2.m           # Verificación del flujo de cámara vía ROS 2
├── viewDetectorROS2.m         # Visualización: cajas + etiquetas + FPS
├── loadYoloDataConfig.m       # Helper: parser del data.yaml (Ultralytics)
├── readYoloLabels.m           # Helper: TXT normalizado -> cajas [x y w h]
├── buildYoloDatastore.m       # Helper: dataset YOLO -> datastores de MATLAB
├── augmentYoloData.m          # Helper: aumento de datos de entrenamiento
├── ensureRGBData.m            # Helper: normalización a 3 canales
├── README_ObjectDetection.md  # Documentación de la suite
└── README_ROS_OD.md           # Preparación de la Jetson (comandos copy-paste)
```

Tópicos publicados: `/detector/image_annotated` (imagen con cajas por clase) y `/detector/detections` (Float32MultiArray N×6: `[x y w h score idClase]`).

## Estructura del workspace de ROS 2

Los nodos se compilan remotamente con **colcon** en workspaces independientes de la Jetson, generados por completo desde MATLAB (sin `CMakeLists.txt` ni C++ manuales):

```
/home/<usuario>/
├── classifier_ws/             # Workspace del nodo de clasificación
│   └── src/classifierjetsonnode/    (paquete ROS 2 generado)
└── detector_ws/               # Workspace del nodo de detección
    └── src/detectorjetsonnode/      (paquete ROS 2 generado)
```

El nombre del paquete queda en minúsculas, pero el **ejecutable conserva las mayúsculas** del entry-point (p. ej. `ros2 run classifierjetsonnode classifierJetsonNode`). Ambos nodos pueden convivir consumiendo el mismo `/image_raw`.

## Requisitos de hardware

| Dispositivo | Requisitos clave |
|---|---|
| **PC Anfitrión** | CPU multicore (Intel Core i7/i9 o AMD Ryzen 7/9), **≥16 GB** de RAM (recomendados 32 GB), almacenamiento SSD y conectividad Ethernet con la Jetson (el video ROS 2 sin comprimir puede saturar WiFi). GPU NVIDIA opcional (acelera el entrenamiento; la compilación del nodo es **remota**, por lo que no se necesita CUDA en el host). |
| **NVIDIA Jetson Orin Nano Super Developer Kit (8 GB)** | Fuente de alimentación oficial, NVMe o microSD **≥64 GB** (UHS-I U3/A2), cámara USB (UVC) y red compartida con el host. Activar el modo MAXN SUPER (`nvpmodel -m 0` + `jetson_clocks`). |

## Requisitos de software

### En el PC host (MATLAB R2025b)

- **Deep Learning Toolbox**, **Computer Vision Toolbox**, **Image Processing Toolbox**.
- **MATLAB Coder**, **GPU Coder** y **ROS Toolbox** (generación y despliegue de nodos ROS 2).
- Add-ons: *GPU Coder Interface for Deep Learning*; *Computer Vision Toolbox Model for YOLO v4 Object Detection* (detección); el modelo base de clasificación elegido (p. ej. *Deep Learning Toolbox Model for ResNet-18 Network*); y, opcional para verificaciones, *MATLAB Coder Support Package for NVIDIA Jetson and NVIDIA DRIVE Platforms*.

### En la Jetson Orin Nano

- **JetPack 7.2** (Jetson Linux L4T r39.2 sobre Ubuntu 24.04, kernel 6.8) con los componentes del SDK instalados vía `sudo apt install nvidia-jetpack`: **CUDA 13.2**, **cuDNN 9.20** y **TensorRT 10.16**.
- **ROS 2 Jazzy Jalisco** (LTS de Ubuntu 24.04) con `ros-jazzy-ament-cmake`, `ros-dev-tools`, colcon y el driver de cámara `ros-jazzy-v4l2-camera`.
- Variables de entorno de CUDA y `source` de ROS insertados **al inicio** de `~/.bashrc` (requisito de la compilación remota por SSH no interactivo).

La preparación completa, con comandos listos para copiar y pegar, verificación por sección y tabla de solución de problemas, está en `Classification/README_jetson_ros2.md` y `ObjectDetection/README_ROS_OD.md`.

## Hallazgos de compatibilidad verificados (JetPack 7.2 + R2025b)

Esta pila es más nueva que la validada oficialmente por MathWorks (JetPack 6.x); el proyecto funciona aplicando estos ajustes, ya incorporados en los scripts:

1. **`ComputeCapability = "8.7"`** en la configuración de GPU Coder: el `nvcc` de CUDA 13 eliminó la arquitectura por defecto (`sm_50`).
2. **Inferencia con cuDNN (FP32)** como ruta validada: TensorRT 10.16 eliminó la API de la capa *fully-connected* que el código generado por R2025b aún utiliza. tiny-YOLOv4 no tiene capas de ese tipo, por lo que TensorRT queda como experimento documentado para la suite de detección.
3. **Tamaños deterministas para codegen**: canales fijados con `I(:,:,1:3)` tras `rosReadImage` y dimensiones fijadas con `imresize` a tamaño constante.
4. **Campo `encoding` explícito** en los mensajes de imagen generados (limitación de `rosWriteImage` en codegen) y restitución defensiva en los visualizadores.
5. **Entorno visible por SSH no interactivo**: variables CUDA y `source` de ROS al inicio de `~/.bashrc` (de lo contrario fallan `nvcc` y `ament_cmake` en la compilación remota).

## Uso y ejecución

1. **Preparar la Jetson** (una sola vez): seguir la guía de la suite correspondiente (`README_jetson_ros2.md` / `README_ROS_OD.md`), secciones 1–10.
2. **Preparar el modelo en el host**: ejecutar `trainClassifier.m`/`trainMNIST.m` o `trainDetector.m` (modos `"train"` o `"export"`); validar localmente con `testClassifier.m`/`testDetector.m`.
3. **Desplegar**: editar IP/credenciales en `deployClassifierJetson.m`/`deployDetectorJetson.m` y ejecutarlo; genera el código CUDA, lo compila en la Jetson con colcon y arranca el nodo.
4. **Ejecutar cada sesión**: lanzar el driver de cámara en la Jetson (`v4l2_camera`, o el servicio systemd opcional), verificar con `ros2 node list` y `ros2 topic hz`, y visualizar desde MATLAB con `viewClassifierROS2.m`/`viewDetectorROS2.m`.

El orden de ejecución detallado (con comandos y verificaciones por paso) está en la sección "Orden de ejecución" del README de cada suite.

## Personalización y entrenamiento de modelos

- **Clasificación**: dataset con particiones predefinidas (`train/`, `valid/` o `val/`, `test/`) y una subcarpeta por clase; `trainClassifier.m` valida la coherencia de clases entre particiones. Modelos externos que solo guardaron `net` se integran con `adaptStudentModel.m`.
- **Detección**: dataset en formato YOLO/Ultralytics (`data.yaml` + anotaciones TXT normalizadas, el export estándar de Roboflow); `buildYoloDatastore.m` imprime un resumen de cajas por clase para detectar desbalance antes de entrenar.

## Créditos

Este proyecto fue desarrollado como parte de un laboratorio de la **Universidad Católica San Pablo** por Diego Banda. Se tomó como fundamento lo expuesto por **Jon Zeosky** y **Sebastian Castro** en su tutorial [`Deep Learning with NVIDIA Jetson and ROS`](https://www.mathworks.com/matlabcentral/fileexchange/69366-deep-learning-with-nvidia-jetson-and-ros?s_eid=PSM_15028), actualizado aquí a la generación Jetson Orin, ROS 2 y el flujo de nodos generados íntegramente desde MATLAB.

## Bibliografía

1. [**NVIDIA JetPack SDK** – página oficial con las versiones de Jetson Linux, CUDA, cuDNN y TensorRT de cada release, incluida la serie JetPack 7.x para Ubuntu 24.04.](https://developer.nvidia.com/embedded/jetpack)
2. [**ROS 2 Documentation: Jazzy Jalisco** – instrucciones oficiales de instalación sobre Ubuntu 24.04 (noble), distribución LTS soportada hasta 2029.](https://docs.ros.org/en/jazzy/Installation.html)
3. [**MathWorks Documentation** – *MATLAB Coder Support Package for NVIDIA Jetson and NVIDIA DRIVE Platforms*: generación remota y despliegue de código MATLAB/Simulink en plataformas NVIDIA.](https://www.mathworks.com/help/coder/nvidia.html)
4. [**MathWorks Documentation** – *ROS Toolbox*: comunicación con redes ROS 2 y generación de nodos ROS 2 independientes desde MATLAB.](https://www.mathworks.com/help/ros/)
5. [**Deep Learning with MATLAB, NVIDIA Jetson, and ROS** – vídeo de MathWorks (Jon Zeosky y Sebastian Castro) que origina el flujo GPU Coder → Jetson → ROS.](https://www.mathworks.com/videos/matlab-and-simulink-robotics-arena-deep-learning-with-nvidia-jetson-and-ros--1542015526909.html)
6. [**MathWorks Miniseries – Object Detection with ROS 2 and Jetson** – episodio que cubre el flujo etiquetado → entrenamiento → generación de código → despliegue como nodo ROS 2 en Jetson.](https://www.youtube.com/watch?v=FHSVW5-W5ew)
7. [**Deploy YOLOv2 to an NVIDIA Jetson** – vídeo de MathWorks sobre generación de código CUDA de detectores YOLO con GPU Coder y despliegue en la GPU embebida.](https://www.youtube.com/watch?v=fD-PKiqYNKo)
