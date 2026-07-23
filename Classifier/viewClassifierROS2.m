%VIEWCLASSIFIERROS2 Visualiza en MATLAB la clasificación que corre en la Jetson.
%
%   La comunicación usa DDS con descubrimiento automático: basta con que
%   el host y la Jetson compartan la misma red y el mismo ROS_DOMAIN_ID.
%   Además del índice de clase, se recibe el vector completo de
%   probabilidades para mostrar el top-3 en pantalla.
%
%   Tópicos consumidos (publicados por classifierJetsonNode en la Jetson):
%       /classifier/image_view (sensor_msgs/Image)
%       /classifier/class_id   (std_msgs/UInt32, índice 1-based)
%       /classifier/scores     (std_msgs/Float32MultiArray)
%
%   Uso:
%       Editar parámetros y ejecutar. Salir con la tecla 'q' o cerrando
%       la ventana.
%
%   Ver también: classifierJetsonNode, deployClassifierJetson

clc; close all;

%% 1. Parámetros de usuario ================================================
modelFile = "trainedClassifier.mat";   % solo para leer classNames
domainId  = 0;                         % debe coincidir con la Jetson

viewTopic   = "/classifier/image_view";
idTopic     = "/classifier/class_id";
scoresTopic = "/classifier/scores";

receiveTimeout = 5;                    % s de espera por imagen

%% 2. Nombres de clase y conexión ROS 2 ===================================
assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
S = load(modelFile, "classNames");
classNames = string(S.classNames(:));

setenv("ROS_DOMAIN_ID", num2str(domainId));
node = ros2node("host_classifier_viewer");
cleanupNode = onCleanup(@() clear("node"));

subView   = ros2subscriber(node, viewTopic,   "sensor_msgs/Image");
subId     = ros2subscriber(node, idTopic,     "std_msgs/UInt32");
subScores = ros2subscriber(node, scoresTopic, "std_msgs/Float32MultiArray");

fprintf("Esperando datos de la Jetson en '%s'...\n", viewTopic);
fprintf(['NOTA: los tópicos /classifier/* solo existen cuando el nodo de\n' ...
         'inferencia está desplegado y corriendo en la Jetson (ejecutar\n' ...
         'antes deployClassifierJetson.m). Verificar con: ros2 node list\n' ...
         '(debe aparecer /classifier_jetson). Revisar también\n' ...
         'ROS_DOMAIN_ID y que ambos equipos compartan la red.\n']);

%% 3. Ventana de visualización ============================================
fig = figure("Name", ...
    "Clasificación en Jetson vía ROS 2  |  presione 'q' para salir", ...
    "NumberTitle", "off");
setappdata(fig, "stop", false);
fig.KeyPressFcn = @(src, evt) setappdata(src, "stop", strcmpi(evt.Key, "q"));
hAx    = axes("Parent", fig);
hImage = [];

%% 4. Bucle de recepción y visualización ==================================
while ishandle(fig) && ~getappdata(fig, "stop")
    [imgMsg, status] = receive(subView, receiveTimeout);
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

    % --- Última clase e índice (con validación de rango) -----------------
    idMsg = subId.LatestMessage;
    if ~isempty(idMsg)
        classNum = double(idMsg.data);
        if classNum >= 1 && classNum <= numel(classNames)
            className = classNames(classNum);
        else
            className = "Índice fuera de rango";
        end
    else
        className = "Esperando dato...";
    end

    % --- Top-3 de probabilidades ------------------------------------------
    topText = "";
    scMsg = subScores.LatestMessage;
    if ~isempty(scMsg) && ~isempty(scMsg.data)
        p = double(scMsg.data(:));
        [ps, order] = sort(p, "descend");
        k = min(3, numel(order));
        lines = classNames(order(1:k)) + ": " + ...
                compose("%.1f %%", 100 * ps(1:k));
        topText = strjoin(lines, newline);
    end

    % --- Actualización eficiente de la figura ----------------------------
    if isempty(hImage) || ~isvalid(hImage)
        hImage = imshow(frame, "Parent", hAx);
    else
        hImage.CData = frame;
    end
    title(hAx, "Clase detectada: " + className, ...
        "FontSize", 14, "Interpreter", "none");
    xlabel(hAx, topText, "FontSize", 11, "Interpreter", "none");
    drawnow limitrate;
end

if ishandle(fig), close(fig); end
fprintf("Visualización finalizada.\n");
