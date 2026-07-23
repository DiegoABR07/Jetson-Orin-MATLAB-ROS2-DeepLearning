function detectorJetsonNode()
%DETECTORJETSONNODE Nodo ROS 2 de detección YOLO para NVIDIA Jetson (entry-point).
%
%   Punto de entrada para GPU Coder: NO se ejecuta interactivamente en
%   MATLAB; se compila y despliega con deployDetectorJetson.m y corre
%   como nodo ROS 2 nativo en la Jetson Orin Nano. Todo el nodo
%   (suscripción a la cámara, inferencia CUDA y publicación) se define en
%   este único archivo; ROS Toolbox y GPU Coder generan a partir de él un
%   paquete ROS 2 estándar que colcon compila en la Jetson.
%
%   Arquitectura:
%
%       /image_raw (sensor_msgs/Image)            <- driver v4l2_camera
%            |
%            v
%       [ este nodo: yolov4ObjectDetector + GPU ]
%            |
%            +--> /detector/image_annotated (sensor_msgs/Image, cajas dibujadas)
%            +--> /detector/detections      (std_msgs/Float32MultiArray)
%
%   Formato de /detector/detections:
%       Vector aplanado de N x 6 valores single, una detección por fila:
%           [x, y, w, h, score, idClase]
%       [x y w h] en píxeles de la imagen publicada en image_annotated
%       (tamaño FRAME_SIZE, 1-based); idClase es el índice 1-based en
%       detector.ClassNames (mismo orden que classNames del MAT). Se usa
%       Float32MultiArray (tipo estándar) para evitar mensajes
%       personalizados; el texto de las etiquetas se añade en el host.
%
%   Requisito del archivo de red:
%       'detectorDeploy.mat' debe existir en la carpeta de trabajo al
%       generar código y contener SOLO la variable 'detector' (lo prepara
%       deployDetectorJetson.m).
%
%   Parámetros de compilación:
%       FRAME_SIZE debe coincidir con el 'image_size' del driver de
%       cámara ([alto ancho]; el driver se lanza con [640,480] ->
%       FRAME_SIZE = [480 640]). El tamaño fijo es requisito de la
%       generación de código; detect() redimensiona internamente a la
%       entrada de la red y devuelve las cajas en coordenadas del cuadro.
%
%   Ver también: deployDetectorJetson, viewDetectorROS2

%#codegen

%% ---- Parámetros de compilación (editar y regenerar si cambian) ----------
IMAGE_TOPIC     = "/image_raw";
ANNOTATED_TOPIC = "/detector/image_annotated";
DET_TOPIC       = "/detector/detections";
NODE_NAME       = "detector_jetson";

THRESHOLD       = 0.50;          % confianza mínima
FRAME_SIZE      = [480 640];     % [alto ancho] = image_size del driver
RECEIVE_TIMEOUT = 5;             % s de espera por cuadro de cámara

% Paleta fija de colores (una fila por clase; se recicla con mod).
% uint8 requerido por insertShape en codegen.
COLORS = uint8([ ...
    230,  57,  70;    % rojo
     42, 157, 143;    % verde azulado
     69, 123, 157;    % azul
    244, 162,  97;    % naranja
    155,  93, 229;    % violeta
    233, 196, 106;    % amarillo
      0, 180, 216;    % celeste
    216,  17,  89]);  % magenta

%% ---- Carga del detector (una sola vez) ----------------------------------
persistent detector
if isempty(detector)
    detector = coder.loadDeepLearningNetwork("detectorDeploy.mat");
end

%% ---- Entidades ROS 2 ----------------------------------------------------
node   = ros2node(NODE_NAME);
sub    = ros2subscriber(node, IMAGE_TOPIC,     "sensor_msgs/Image");
pubImg = ros2publisher(node,  ANNOTATED_TOPIC, "sensor_msgs/Image");
pubDet = ros2publisher(node,  DET_TOPIC,       "std_msgs/Float32MultiArray");

%% ---- Bucle principal: capturar -> detectar -> publicar ------------------
while true
    % 'status' evita que un timeout detenga el nodo (p. ej. si el driver
    % de cámara aún no publica).
    [imgMsg, status] = receive(sub, RECEIVE_TIMEOUT);
    if ~status
        continue
    end

    % Codificación fijada en compilación. El driver v4l2_camera publica
    % rgb8 (parámetro 'output_encoding').
    I = rosReadImage(imgMsg, "Encoding", "rgb8");

    % rosReadImage devuelve TAMAÑO VARIABLE (las dimensiones llegan en el
    % mensaje). El generador de código exige dimensiones deterministas:
    % la indexación con rango constante (1:3) fija los canales (rgb8
    % garantiza 3) y el imresize a tamaño constante fija alto y ancho.
    I = I(:, :, 1:3);
    I = imresize(I, coder.const(FRAME_SIZE));

    % ---- Inferencia (GPU en el ejecutable generado) ----------------------
    [bboxes, scores, labels] = detect(detector, I, "Threshold", THRESHOLD);
    numDet = size(bboxes, 1);

    if numDet > 0
        % double(categorical) devuelve el índice 1-based de la categoría,
        % consistente con el orden de detector.ClassNames.
        classIdx  = double(labels);
        boxColors = COLORS(mod(classIdx - 1, size(COLORS, 1)) + 1, :);

        Iout = insertShape(I, "rectangle", bboxes, ...
            "LineWidth", 3, "Color", boxColors);

        detRows = single([bboxes, scores(:), classIdx(:)]);  % N x 6
        detVec  = reshape(detRows.', [], 1);                 % fila a fila
    else
        Iout   = I;
        detVec = zeros(0, 1, "single");
    end

    % ---- Publicación de la imagen anotada --------------------------------
    outImg = ros2message("sensor_msgs/Image");
    outImg = rosWriteImage(outImg, Iout, "Encoding", "rgb8");
    % El rosWriteImage generado puede no rellenar el campo 'encoding'
    % (llegaría vacío al host y rosReadImage lo rechazaría); se asigna
    % explícitamente para que el mensaje salga completo de origen.
    outImg.encoding = 'rgb8';
    outImg.header = imgMsg.header;      % conservar timestamp de origen
    send(pubImg, outImg);

    % ---- Publicación de las detecciones numéricas ------------------------
    detMsg = ros2message("std_msgs/Float32MultiArray");
    detMsg.data = detVec;
    send(pubDet, detMsg);
end
end
