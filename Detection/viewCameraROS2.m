%VIEWCAMERAROS2 Visor simple de un tópico de imagen ROS 2 (prueba de cámara).
%
%   Sirve para verificar que el driver de cámara de la Jetson
%   (v4l2_camera publicando en /image_raw) llega correctamente al host
%   ANTES de desplegar el nodo de clasificación. También puede apuntarse
%   a /classifier/image_view.
%
%   La comunicación usa DDS con descubrimiento automático (misma red y
%   mismo ROS_DOMAIN_ID que la Jetson). Salir con la tecla 'q' o
%   cerrando la ventana.
%
%   Ver también: viewClassifierROS2

clc; close all;

%% 1. Parámetros de usuario ================================================
domainId       = 0;              % debe coincidir con la Jetson
imageTopic     = "/image_raw";   % o "/classifier/image_view"
receiveTimeout = 5;              % s de espera por imagen

%% 2. Conexión ROS 2 =======================================================
setenv("ROS_DOMAIN_ID", num2str(domainId));
node = ros2node("host_camera_viewer");
cleanupNode = onCleanup(@() clear("node"));

sub = ros2subscriber(node, imageTopic, "sensor_msgs/Image");
fprintf("Esperando imágenes en '%s'...\n", imageTopic);

%% 3. Bucle de visualización ==============================================
fig = figure("Name", "Cámara ROS 2  |  presione 'q' para salir", ...
    "NumberTitle", "off");
setappdata(fig, "stop", false);
fig.KeyPressFcn = @(src, evt) setappdata(src, "stop", strcmpi(evt.Key, "q"));
hImage = [];

while ishandle(fig) && ~getappdata(fig, "stop")
    [imgMsg, status] = receive(sub, receiveTimeout);
    if ~status
        continue
    end
    frame = rosReadImage(imgMsg);

    if isempty(hImage) || ~isvalid(hImage)
        hImage = imshow(frame);
        title("Cámara en vivo (ROS 2)");
    else
        hImage.CData = frame;
    end
    drawnow limitrate;
end

if ishandle(fig), close(fig); end
fprintf("Visor de cámara finalizado.\n");
