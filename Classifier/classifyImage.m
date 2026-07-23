function [classIdx, className, score, allScores] = classifyImage(im, modelFile)
%CLASSIFYIMAGE Clasifica una imagen con un modelo entrenado (ejecución en host).
%
%   [classIdx, className, score, allScores] = classifyImage(im, modelFile)
%
%   Opera sobre dlnetwork (el tipo de red que devuelven
%   imagePretrainedNetwork y trainnet) y usa el MAT estandarizado que
%   produce trainClassifier.m (variables: net, classNames, inputSize).
%
%   Entradas:
%       im        - Imagen RGB o en escala de grises (uint8 recomendado),
%                   de cualquier tamaño.
%       modelFile - (opcional) Ruta al .mat del modelo.
%                   Por defecto: "trainedClassifier.mat".
%
%   Salidas:
%       classIdx  - Índice 1-based de la clase con mayor probabilidad.
%       className - Nombre de la clase (string).
%       score     - Probabilidad de la clase ganadora (softmax).
%       allScores - Vector con las probabilidades de todas las clases.
%
%   Notas de implementación:
%       * La red se carga UNA sola vez (variables persistentes) y se
%         recarga automáticamente si cambia modelFile.
%       * IMPORTANTE (rango de píxeles): la red se alimenta con
%         single(im) en rango 0-255, SIN im2single. La normalización
%         (zerocenter/zscore) está integrada en la capa de entrada del
%         dlnetwork y sus estadísticas se calcularon en escala 0-255;
%         escalar a [0,1] antes degradaría la precisión.
%       * El preprocesamiento (canales + imresize directo a inputSize)
%         replica el del entrenamiento y el del nodo Jetson.
%
%   Ver también: trainClassifier, testClassifier, classifierJetsonNode

arguments
    im        {mustBeNonempty}
    modelFile (1,1) string = "trainedClassifier.mat"
end

%% ---- Carga persistente del modelo --------------------------------------
persistent netP classNamesP inputSizeP fileP

if isempty(netP) || isempty(fileP) || fileP ~= modelFile
    assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
    S = load(modelFile, "net", "classNames", "inputSize");
    netP        = S.net;
    classNamesP = string(S.classNames(:));
    inputSizeP  = S.inputSize;
    fileP       = modelFile;
end

%% ---- Preprocesamiento (idéntico a entrenamiento y despliegue) ----------
C = inputSizeP(3);
if C == 1 && size(im, 3) == 3
    imProc = rgb2gray(im);
elseif C == 3 && size(im, 3) == 1
    imProc = repmat(im, [1, 1, 3]);
else
    imProc = im;
end

imResized = imresize(imProc, inputSizeP(1:2));

%% ---- Inferencia ---------------------------------------------------------
% dlnetwork requiere entrada dlarray con formato: "SSC" = Spatial,
% Spatial, Channel (una sola imagen, sin dimensión de batch).
dlX    = dlarray(single(imResized), "SSC");
dlOut  = predict(netP, dlX);
allScores = extractdata(dlOut);
allScores = double(allScores(:));

[score, classIdx] = max(allScores);
className = classNamesP(classIdx);
end
