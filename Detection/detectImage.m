function [bboxes, scores, labels, annotated] = detectImage(im, modelFile, threshold)
%DETECTIMAGE Detecta objetos en una imagen con un modelo entrenado (host).
%
%   [bboxes, scores, labels, annotated] = detectImage(im, modelFile, threshold)
%
%   Análoga a classifyImage.m de la suite de clasificación: opera sobre
%   el yolov4ObjectDetector del MAT estandarizado que produce
%   trainDetector.m (variables: detector, classNames, inputSize).
%
%   Entradas:
%       im        - Imagen RGB o en escala de grises, de cualquier tamaño.
%       modelFile - (opcional) Ruta al .mat. Def.: "trainedDetector.mat".
%       threshold - (opcional) Confianza mínima. Def.: 0.5.
%
%   Salidas:
%       bboxes    - M x 4 [x y w h] en píxeles de la imagen original.
%       scores    - M x 1 confianzas.
%       labels    - M x 1 etiquetas categóricas.
%       annotated - Imagen con las detecciones dibujadas (para mostrar).
%
%   Notas:
%       * El detector se carga UNA sola vez (variables persistentes) y se
%         recarga automáticamente si cambia modelFile.
%       * detect() aplica internamente el redimensionado a la entrada de
%         la red y devuelve las cajas en coordenadas de la imagen
%         original, por lo que no se requiere preprocesamiento manual
%         (solo asegurar 3 canales).
%
%   Ver también: trainDetector, testDetector, detectorJetsonNode

arguments
    im        {mustBeNonempty}
    modelFile (1,1) string = "trainedDetector.mat"
    threshold (1,1) double {mustBeInRange(threshold, 0, 1)} = 0.5
end

%% ---- Carga persistente del modelo --------------------------------------
persistent detP fileP

if isempty(detP) || isempty(fileP) || fileP ~= modelFile
    assert(isfile(modelFile), "No se encontró el modelo: %s", modelFile);
    S = load(modelFile, "detector");
    detP  = S.detector;
    fileP = modelFile;
end

%% ---- Preprocesamiento mínimo (canales) ----------------------------------
if size(im, 3) == 1
    im = repmat(im, [1, 1, 3]);
elseif size(im, 3) > 3
    im = im(:, :, 1:3);
end

%% ---- Detección -----------------------------------------------------------
[bboxes, scores, labels] = detect(detP, im, "Threshold", threshold);

%% ---- Imagen anotada (salida opcional) ------------------------------------
if nargout >= 4
    if isempty(bboxes)
        annotated = im;
    else
        txt = string(labels) + " " + compose("%.2f", scores);
        annotated = insertObjectAnnotation(im, "rectangle", bboxes, ...
            cellstr(txt), "LineWidth", 3, "FontSize", 14);
    end
end
end
