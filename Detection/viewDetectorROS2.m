%VIEWDETECTORROS2 Visualiza en MATLAB la detección YOLO que corre en la Jetson.
%
%   La comunicación usa DDS con descubrimiento automático: basta con que
%   el host y la Jetson compartan la misma red y el mismo ROS_DOMAIN_ID.
%   La imagen llega con las cajas ya dibujadas desde la Jetson; el host
%   superpone el texto (clase + confianza) desde el tópico de
%   detecciones, que es costoso de generar en código embebido.
%
%   Tópicos consumidos (publicados por detectorJetsonNode en la Jetson):
%       /detector/image_annotated (sensor_msgs/Image)
%       /detector/detections      (std_msgs/Float32MultiArray, N x 6:
%                                  [x y w h score idClase] por detección)
%
%   Uso:
%       Editar parámetros y ejecutar. Salir con la tecla 'q' o cerrando
%       la ventana.
%
%   Ver también: detectorJetsonNode, deployDetectorJetson

clc; close all;

%% 1. Parámetros de usuario ================================================
modelFile = "trainedDetector.mat";     % solo para leer classNames
domainId  = 0;                         % debe coincidir con la Jetson

imageTopic     = "/detector/image_annotated";
detectionTopic = "/detector/detections";

receiveTimeout = 5;                    % s de espera por imagen

%% 2. Nombres de clase y conexión ROS 2 ===================================
assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
S = load(modelFile, "classNames");
classNames = string(S.classNames(:));

setenv("ROS_DOMAIN_ID", num2str(domainId));
node = ros2node("host_detector_viewer");
cleanupNode = onCleanup(@() clear("node"));

subImg = ros2subscriber(node, imageTopic,     "sensor_msgs/Image");
subDet = ros2subscriber(node, detectionTopic, "std_msgs/Float32MultiArray");

fprintf("Esperando datos de la Jetson en '%s'...\n", imageTopic);
fprintf(['NOTA: los tópicos /detector/* solo existen cuando el nodo de\n' ...
         'inferencia está desplegado y corriendo en la Jetson (ejecutar\n' ...
         'antes deployDetectorJetson.m). Verificar con: ros2 node list\n' ...
         '(debe aparecer /detector_jetson). Revisar también\n' ...
         'ROS_DOMAIN_ID y que ambos equipos compartan la red.\n']);

%% 3. Ventana de visualización ============================================
fig = figure("Name", ...
    "Detección YOLO en Jetson vía ROS 2  |  presione 'q' para salir", ...
    "NumberTitle", "off");
setappdata(fig, "stop", false);
fig.KeyPressFcn = @(src, evt) setappdata(src, "stop", strcmpi(evt.Key, "q"));
hAx    = axes("Parent", fig);
hImage = [];

%% 4. Bucle de recepción y visualización ==================================
tPrev = tic;
while ishandle(fig) && ~getappdata(fig, "stop")
    [imgMsg, status] = receive(subImg, receiveTimeout);
    if ~status
        continue    % sin datos aún; seguir esperando
    end
    % El campo 'encoding' puede llegar vacío desde el código C++ generado
    % (limitación de rosWriteImage en codegen); el nodo siempre publica
    % rgb8, así que se restituye antes de decodificar.
    if isempty(imgMsg.encoding)
        imgMsg.encoding = 'rgb8';
    end
    frame = rosReadImage(imgMsg);

    % --- Etiquetas desde el último mensaje de detecciones -----------------
    numDet = 0;
    detMsg = subDet.LatestMessage;
    if ~isempty(detMsg) && ~isempty(detMsg.data)
        D = reshape(double(detMsg.data), 6, []).';   % N x [x y w h s id]
        numDet = size(D, 1);

        % Validación de rango del índice de clase antes de indexar.
        ok  = D(:, 6) >= 1 & D(:, 6) <= numel(classNames);
        D   = D(ok, :);
        if ~isempty(D)
            txt = classNames(D(:, 6)) + " " + compose("%.2f", D(:, 5));
            pos = max(D(:, 1:2) - [0, 22], 1);       % texto sobre cada caja
            frame = insertText(frame, pos, cellstr(txt), ...
                "FontSize", 14, "BoxColor", "black", "TextColor", "white");
        end
    end

    % --- HUD con FPS de recepción y nº de detecciones ---------------------
    fps = 1 / max(toc(tPrev), eps);
    tPrev = tic;
    frame = insertText(frame, [10 10], ...
        sprintf("FPS (red): %.1f | Detecciones: %d", fps, numDet), ...
        "FontSize", 16, "BoxColor", "black", "TextColor", "white");

    % --- Actualización eficiente de la figura -----------------------------
    if isempty(hImage) || ~isvalid(hImage)
        hImage = imshow(frame, "Parent", hAx);
    else
        hImage.CData = frame;
    end
    drawnow limitrate;
end

if ishandle(fig), close(fig); end
fprintf("Visualización finalizada.\n");
