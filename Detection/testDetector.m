%TESTDETECTOR Prueba local (host) del detector entrenado.
%
%   Carga una imagen, la procesa con detectImage.m y muestra la imagen
%   anotada junto con la tabla de detecciones en consola (útil para
%   validar el modelo antes de desplegar en la Jetson).
%
%   Uso:
%       Editar 'imgPath' (y opcionalmente 'modelFile' y 'threshold') y
%       ejecutar.
%
%   Ver también: detectImage, trainDetector

clc; close all;
clear detectImage                      % fuerza recarga del modelo persistente

%% 1. Parámetros de usuario ================================================
modelFile = "trainedDetector.mat";
imgPath   = "Samples/escena.jpg";      % imagen de prueba (editar)
threshold = 0.5;                       % confianza mínima

%% 2. Detección ============================================================
assert(isfile(imgPath), "No se encontró la imagen: %s", imgPath);
im = imread(imgPath);

[bboxes, scores, labels, annotated] = detectImage(im, modelFile, threshold);

%% 3. Resultados en consola ================================================
fprintf("Imagen: %s\n", imgPath);
fprintf("Detecciones (umbral %.2f): %d\n\n", threshold, size(bboxes, 1));
if ~isempty(bboxes)
    T = table(string(labels), scores, bboxes(:, 1), bboxes(:, 2), ...
        bboxes(:, 3), bboxes(:, 4), "VariableNames", ...
        {'Clase', 'Confianza', 'x', 'y', 'ancho', 'alto'});
    disp(T);
else
    fprintf("Sin detecciones sobre el umbral. Pruebe con un umbral menor\n");
    fprintf("o verifique que la imagen contiene las clases del modelo.\n");
end

%% 4. Visualización ========================================================
figure("Name", "Prueba del detector");
imshow(annotated);
title(sprintf("%d detecciones (umbral %.2f)", size(bboxes, 1), threshold));
