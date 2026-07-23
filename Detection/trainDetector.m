%TRAINDETECTOR Entrenamiento local o exportación directa de un detector YOLO.
%
%   Dos modos de operación (parámetro 'mode' en la sección 1):
%
%     mode = "export" : Exporta tiny-YOLOv4 preentrenado TAL CUAL, con sus
%                       80 clases de COCO, a trainedDetector.mat. No
%                       requiere dataset. Útil para demos con objetos
%                       cotidianos (persona, botella, taza, laptop...) y
%                       para validar el flujo completo de despliegue.
%
%     mode = "train"  : Transferencia de aprendizaje sobre un dataset
%                       propio en formato YOLO/Ultralytics (data.yaml +
%                       anotaciones TXT), con evaluación (mAP) integrada.
%
%   Dataset esperado (solo modo "train"; formato Ultralytics/Roboflow):
%       datasetDir/
%       ├── data.yaml            (path, train, val, test, nc, names)
%       ├── train/images/*.jpg   +  train/labels/*.txt
%       ├── valid/... o val/...  (declararlo en el YAML: val: valid)
%       └── test/...             (opcional)
%
%   Arquitectura: tiny-YOLOv4 (Computer Vision Toolbox), equivalente
%   ligero a YOLOv5n con soporte nativo de entrenamiento y de generación
%   de código CUDA, requisito del despliegue en la Jetson.
%
%   Salida (contrato del pipeline de detección; lo consumen testDetector,
%   deployDetectorJetson y viewDetectorROS2):
%       trainedDetector.mat con: detector, classNames, inputSize,
%       baseModel, mode, info y testMetrics ([] en modo "export").
%
%   Requisitos:
%       * Deep Learning Toolbox, Computer Vision Toolbox.
%       * Add-on "Computer Vision Toolbox Model for YOLO v4 Object
%         Detection" (para el modelo base "tiny-yolov4-coco").
%
%   Ver también: detectImage, testDetector, deployDetectorJetson

clear; clc; close all;
rng(0);                                   % reproducibilidad

%% 1. Parámetros de usuario ================================================
mode       = "train";                     % "train" | "export"
outputFile = "trainedDetector.mat";

% --- Solo modo "train" ---------------------------------------------------
dataConfigFile = "data.yaml";             % YAML del dataset (editar ruta)
inputSize      = [416 416 3];             % entrada del detector
numAnchors     = 6;                       % tiny-YOLOv4: 2 cabezas x 3 anchors

maxEpochs        = 80;
miniBatchSize    = 8;                     % reducir si falta memoria
initialLearnRate = 1e-3;

evalThreshold    = 0.01;                  % umbral bajo p/ métricas correctas
displayThreshold = 0.50;                  % umbral del mosaico cualitativo

%% 2. Ejecución según el modo =============================================
switch mode
    %% ===================== MODO "export" ================================
    case "export"
        fprintf("Modo 'export': exportando tiny-YOLOv4 preentrenado (COCO).\n");

        detector   = yolov4ObjectDetector("tiny-yolov4-coco");
        classNames = string(detector.ClassNames);
        inputSize  = detector.InputSize;

        info        = [];
        testMetrics = [];
        baseModel   = "tiny-yolov4-coco";

        fprintf("Detector '%s' | entrada [%d %d %d] | %d clases (COCO).\n", ...
            baseModel, inputSize, numel(classNames));

    %% ===================== MODO "train" =================================
    case "train"
        % ---- 2.1 Configuración y datastores ------------------------------
        cfg = loadYoloDataConfig(dataConfigFile);
        classNames = cfg.names;
        fprintf("Dataset: %s | %d clases: %s\n", ...
            cfg.root, cfg.nc, strjoin(classNames, ", "));

        [dsTrain, ~, bldsTrain, statsTrain] = buildYoloDatastore(cfg, "train");
        [dsVal] = buildYoloDatastore(cfg, "val");

        dsTrainAug = transform(dsTrain, @augmentYoloData);
        dsValRGB   = transform(dsVal,   @ensureRGBData);

        % ---- 2.2 Anchor boxes (k-means sobre IoU) -------------------------
        % Los de mayor área van a la primera cabeza de detección.
        [anchors, meanIoU] = estimateAnchorBoxes(bldsTrain, numAnchors);
        [~, sortIdx] = sort(anchors(:, 1) .* anchors(:, 2), "descend");
        anchors = anchors(sortIdx, :);
        anchorBoxes = {anchors(1:3, :); anchors(4:6, :)};
        fprintf("Anchors estimados (IoU medio = %.3f).\n", meanIoU);

        % ---- 2.3 Detector base (transferencia) ----------------------------
        baseModel = "tiny-yolov4-coco";
        detector = yolov4ObjectDetector(baseModel, classNames, anchorBoxes, ...
            "InputSize", inputSize);

        % ---- 2.4 Entrenamiento --------------------------------------------
        iterPerEpoch = max(1, floor(statsTrain.numImagesUsed / miniBatchSize));
        options = trainingOptions("adam", ...
            "InitialLearnRate",     initialLearnRate, ...
            "LearnRateSchedule",    "piecewise", ...
            "LearnRateDropFactor",  0.5, ...
            "LearnRateDropPeriod",  round(maxEpochs / 3), ...
            "L2Regularization",     5e-4, ...
            "MaxEpochs",            maxEpochs, ...
            "MiniBatchSize",        miniBatchSize, ...
            "Shuffle",              "every-epoch", ...
            "ValidationData",       dsValRGB, ...
            "ValidationFrequency",  iterPerEpoch, ...
            "ExecutionEnvironment", "auto", ...
            "Plots",                "training-progress", ...
            "VerboseFrequency",     20);

        fprintf("Entrenando (%d épocas, batch %d)...\n", maxEpochs, miniBatchSize);
        [detector, info] = trainYOLOv4ObjectDetector(dsTrainAug, detector, options);

        % ---- 2.5 Evaluación (test; si no existe, val con aviso) ----------
        if strlength(cfg.test) > 0 && isfolder(fullfile(cfg.root, cfg.test))
            evalSplit = "test";
        else
            warning(['No hay partición ''test''; las métricas se calculan ' ...
                'sobre ''val'' (interpretarlas con cautela).']);
            evalSplit = "val";
        end
        [~, imdsEval, bldsEval, statsEval] = buildYoloDatastore(cfg, evalSplit);

        fprintf("Evaluando sobre %d imágenes de '%s'...\n", ...
            statsEval.numImagesUsed, evalSplit);
        results = detect(detector, imdsEval, ...
            "MiniBatchSize", 8, "Threshold", evalThreshold);

        % mAP@0.5 (PASCAL VOC) y mAP@[.5:.05:.95] (COCO). UNIQUE evita
        % umbrales duplicados (verificado: los repetidos son error).
        overlapThresholds = unique([0.5, 0.5:0.05:0.95]);
        testMetrics = evaluateObjectDetection(results, bldsEval, ...
            overlapThresholds);

        fprintf("\n===== Métricas del dataset (%s) =====\n", evalSplit);
        disp(testMetrics.DatasetMetrics);
        fprintf("===== Métricas por clase =====\n");
        disp(testMetrics.ClassMetrics);

        % ---- 2.6 Mosaico cualitativo --------------------------------------
        n = min(6, numel(imdsEval.Files));
        sampleIdx = randperm(numel(imdsEval.Files), n);
        tiles = cell(1, n);
        for k = 1:n
            I = imread(imdsEval.Files{sampleIdx(k)});
            if size(I, 3) == 1, I = repmat(I, [1 1 3]); end
            [bxs, scr, lbl] = detect(detector, I, ...
                "Threshold", displayThreshold);
            if ~isempty(bxs)
                txt = string(lbl) + " " + compose("%.2f", scr);
                I = insertObjectAnnotation(I, "rectangle", bxs, ...
                    cellstr(txt), "LineWidth", 3, "FontSize", 14);
            end
            tiles{k} = I;
        end
        figQ = figure("Name", "Detecciones de ejemplo");
        montage(tiles, "BorderSize", 4);
        title(sprintf("Detecciones en '%s' (umbral = %.2f)", ...
            evalSplit, displayThreshold));
        exportgraphics(figQ, "detector_qualitative.png", "Resolution", 150);

    otherwise
        error("Modo no válido: '%s'. Use ""train"" o ""export"".", mode);
end

%% 3. Guardado (contrato común a ambos modos) =============================
save(outputFile, "detector", "classNames", "inputSize", "baseModel", ...
    "mode", "info", "testMetrics");

fprintf("\nDetector guardado en '%s' (modo '%s').\n", outputFile, mode);
fprintf("Siguiente paso: testDetector.m (prueba local) y luego\n");
fprintf("deployDetectorJetson.m (despliegue en la Jetson).\n");
