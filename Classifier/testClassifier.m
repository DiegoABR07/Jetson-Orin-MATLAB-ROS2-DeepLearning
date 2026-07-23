%TESTCLASSIFIER Prueba local (host) del clasificador entrenado.
%
%   Carga una imagen, la clasifica
%   con classifyImage.m y muestra la imagen con la clase ganadora y el
%   top-3 de probabilidades (útil para detectar clases confundidas antes
%   de desplegar).
%
%   Uso:
%       Editar 'imgPath' (y opcionalmente 'modelFile') y ejecutar.
%
%   Ver también: classifyImage, trainClassifier

clc; close all;
clear classifyImage                    % fuerza recarga del modelo persistente

%% 1. Parámetros de usuario ================================================
modelFile = "trainedClassifier.mat";
imgPath   = "Samples/peppers.jpg";     % imagen de prueba (editar)

%% 2. Clasificación ========================================================
assert(isfile(imgPath), "No se encontró la imagen: %s", imgPath);
im = imread(imgPath);

[classIdx, className, score, allScores] = classifyImage(im, modelFile);

%% 3. Top-3 de probabilidades =============================================
S = load(modelFile, "classNames");
classNames = string(S.classNames(:));

[sortedScores, order] = sort(allScores, "descend");
k = min(3, numel(order));

fprintf("Imagen: %s\n", imgPath);
fprintf("Clase predicha: %s (índice %d, prob. %.1f %%)\n\n", ...
    className, classIdx, 100 * score);
fprintf("Top-%d:\n", k);
for i = 1:k
    fprintf("  %-20s %6.2f %%\n", classNames(order(i)), 100 * sortedScores(i));
end

%% 4. Visualización ========================================================
figure("Name", "Prueba del clasificador");
imshow(im);
title(sprintf("%s  (%.1f %%)", className, 100 * score), ...
    "Interpreter", "none", "FontSize", 14);
