function classifierJetsonNode()
%CLASSIFIERJETSONNODE Nodo ROS 2 de clasificación para NVIDIA Jetson (entry-point).
%
%   Punto de entrada para GPU Coder: NO se ejecuta interactivamente en
%   MATLAB; se compila y despliega con deployClassifierJetson.m y corre
%   como nodo ROS 2 nativo en la Jetson Orin Nano.
%
%   Todo el nodo (suscripción a la cámara, preprocesamiento, inferencia
%   CUDA/TensorRT y publicación de resultados) se define en este único
%   archivo; ROS Toolbox y GPU Coder generan a partir de él un paquete
%   ROS 2 estándar que colcon compila en la Jetson.
%
%   Arquitectura:
%
%       /image_raw (sensor_msgs/Image)            <- driver v4l2_camera
%            |
%            v
%       [ este nodo: dlnetwork + TensorRT (GPU) ]
%            |
%            +--> /classifier/class_id   (std_msgs/UInt32, índice 1-based)
%            +--> /classifier/scores     (std_msgs/Float32MultiArray)
%            +--> /classifier/image_view (sensor_msgs/Image, entrada de la red)
%
%   El tópico /classifier/scores publica las probabilidades de TODAS las
%   clases, lo que permite mostrar un top-k en el host sin tráfico extra.
%
%   IMPORTANTE - INPUT_SIZE:
%       Debe coincidir con net.Layers(1).InputSize del modelo desplegado
%       (deployClassifierJetson.m lo imprime y verifica). Es constante de
%       compilación porque TensorRT requiere tamaños de entrada fijos.
%
%   Requisito del archivo de red:
%       'classifierDeploy.mat' debe existir en la carpeta de trabajo al
%       generar código y contener SOLO la variable 'net' (lo prepara
%       deployClassifierJetson.m).
%
%   Ver también: deployClassifierJetson, viewClassifierROS2

%#codegen

%% ---- Parámetros de compilación (editar y regenerar si cambian) ----------
IMAGE_TOPIC  = "/image_raw";
ID_TOPIC     = "/classifier/class_id";
SCORES_TOPIC = "/classifier/scores";
VIEW_TOPIC   = "/classifier/image_view";
NODE_NAME    = "classifier_jetson";

INPUT_SIZE      = [28 28 1];   % <- igualar a net.Layers(1).InputSize
RECEIVE_TIMEOUT = 5;             % s de espera por cuadro de cámara

%% ---- Carga de la red (una sola vez) ------------------------------------
persistent net
if isempty(net)
    net = coder.loadDeepLearningNetwork("classifierDeploy.mat");
end

%% ---- Entidades ROS 2 ----------------------------------------------------
node      = ros2node(NODE_NAME);
sub       = ros2subscriber(node, IMAGE_TOPIC,  "sensor_msgs/Image");
pubId     = ros2publisher(node,  ID_TOPIC,     "std_msgs/UInt32");
pubScores = ros2publisher(node,  SCORES_TOPIC, "std_msgs/Float32MultiArray");
pubView   = ros2publisher(node,  VIEW_TOPIC,   "sensor_msgs/Image");

%% ---- Bucle principal: capturar -> clasificar -> publicar ----------------
while true
    % 'status' evita que un timeout detenga el nodo (p. ej. si el driver
    % de cámara aún no publica).
    [imgMsg, status] = receive(sub, RECEIVE_TIMEOUT);
    if ~status
        continue
    end

    % Codificación fijada en compilación (canales deterministas). El
    % driver v4l2_camera publica rgb8 (parámetro 'output_encoding').
    I = rosReadImage(imgMsg, "Encoding", "rgb8");

    % rosReadImage devuelve una imagen de TAMAÑO VARIABLE (las
    % dimensiones llegan dentro del mensaje en tiempo de ejecución), por
    % lo que el generador de código no conoce el nº de canales. PREDICT
    % exige canales constantes en compilación (TensorRT necesita la
    % forma completa del tensor). La codificación rgb8 garantiza 3
    % canales; la indexación con rango constante (1:3) fija esa
    % dimensión para el generador de código.
    I = I(:, :, 1:3);

    % ---- Preprocesamiento: MISMO que en entrenamiento/host --------------
    if INPUT_SIZE(3) == 1
        Iproc = rgb2gray(I);
    else
        Iproc = I;
    end
    imResized = imresize(Iproc, coder.const(INPUT_SIZE(1:2)));

    % ---- Inferencia (TensorRT en el ejecutable generado) ----------------
    % La red se alimenta en rango 0-255 (single); la normalización está
    % integrada en la capa de entrada del dlnetwork.
    dlX    = dlarray(single(imResized), "SSC");
    scores = extractdata(predict(net, dlX));
    scores = single(scores(:));
    [~, classIdx] = max(scores);

    % ---- Publicación del índice de clase --------------------------------
    idMsg = ros2message("std_msgs/UInt32");
    idMsg.data = uint32(classIdx);
    send(pubId, idMsg);

    % ---- Publicación del vector completo de probabilidades --------------
    scMsg = ros2message("std_msgs/Float32MultiArray");
    scMsg.data = scores;
    send(pubScores, scMsg);

    % ---- Publicación de la imagen de entrada a la red (depuración) ------
    if INPUT_SIZE(3) == 1
        Iview = repmat(imResized, [1, 1, 3]);   % rgb8 para visualización
    else
        Iview = imResized;
    end
    viewMsg = ros2message("sensor_msgs/Image");
    viewMsg = rosWriteImage(viewMsg, Iview, "Encoding", "rgb8");
    % El rosWriteImage generado puede no rellenar el campo 'encoding'
    % (llegaría vacío al host y rosReadImage lo rechazaría); se asigna
    % explícitamente para que el mensaje salga completo de origen.
    viewMsg.encoding = 'rgb8';
    viewMsg.header = imgMsg.header;             % conservar timestamp origen
    send(pubView, viewMsg);
end
end
