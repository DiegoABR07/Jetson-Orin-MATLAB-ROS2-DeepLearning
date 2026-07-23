%TRAINCLASSIFIER Entrenamiento local o exportación directa de un clasificador.
%
%   Dos modos de operación (parámetro 'mode' en la sección 1):
%
%     mode = "export" : Exporta la red base preentrenada POR DEFECTO, con 
%                       TODAS las clases de ImageNet, a trainedClassifier.mat.
%                       No requiere dataset.
%
%     mode = "train"  : Transferencia de aprendizaje sobre un dataset
%                       propio con particiones predefinidas (ver abajo).
%
%   Dataset esperado (solo modo "train"; particiones predefinidas, estilo
%   Roboflow; las etiquetas se toman del nombre de la subcarpeta de clase):
%       datasetDir/
%       ├── train/                  (obligatoria)
%       │   ├── clase_1/*.jpg|png|bmp
%       │   └── clase_2/...
%       ├── valid/                  (o "val"; opcional)
%       │   ├── clase_1/...
%       │   └── clase_2/...
%       └── test/                   (opcional)
%           └── ...
%
%   Comportamiento ante particiones ausentes:
%       * Sin valid/val -> se separa automáticamente un porcentaje de
%         train (autoValFraction) como validación, con aviso.
%       * Sin test      -> las métricas finales se calculan sobre la
%         partición de validación, con aviso (interpretarlas con cautela:
%         la validación también guio el entrenamiento).
%
%   Salida (La misma en ambos modos):
%       trainedClassifier.mat con: net (dlnetwork), classNames, inputSize,
%       baseModel, mode, info y testAccuracy (NaN en modo "export").
%
%   Requisitos:
%       * Deep Learning Toolbox.
%       * Add-on del modelo base elegido (p. ej. "Deep Learning Toolbox
%         Model for ResNet-18 Network").
%
%   Nota sobre el preprocesamiento (IMPORTANTE):
%       Las imágenes se redimensionan directamente a inputSize(1:2)
%       (augmentedImageDatastore). El MISMO redimensionado se aplica en
%       classifyImage.m y en classifierJetsonNode.m: mantener el
%       preprocesamiento idéntico entre entrenamiento e inferencia es
%       clave para que la precisión desplegada coincida con la medida.
%
%   Ver también: classifyImage, testClassifier, deployClassifierJetson

clear; clc; close all;
rng(0);                                   % reproducibilidad

%% 1. Parámetros de usuario ================================================
mode       = "export";                     % "train" | "export"
baseModel  = "resnet18";                  % "squeezenet" | "resnet18" |
                                          % "mobilenetv2" | "googlenet" ...
outputFile = "trainedClassifier.mat";

% --- Solo modo "train" ---------------------------------------------------
datasetDir      = "MiDatasetClasificacion"; % raíz con train/valid/test
autoValFraction = 0.15;                     % si no existe valid/val

maxEpochs        = 15;
miniBatchSize    = 32;                    % reducir si falta memoria
initialLearnRate = 1e-4;                  % bajo: transferencia de aprendizaje

%% 2. Ejecución según el modo =============================================
switch mode
    %% ===================== MODO "export" ================================
    case "export"
        fprintf("Modo 'export': exportando '%s' preentrenada (ImageNet).\n", ...
            baseModel);

        % Sin "NumClasses", devuelve la red original y sus clases ImageNet.
        [net, rawNames] = imagePretrainedNetwork(baseModel);
        classNames = string(rawNames(:));
        inputSize  = net.Layers(1).InputSize;

        info         = [];                % no hay entrenamiento
        testAccuracy = NaN;

        fprintf("Red '%s' | entrada [%d %d %d] | %d clases (ImageNet).\n", ...
            baseModel, inputSize, numel(classNames));

    %% ===================== MODO "train" =================================
    case "train"
        % ---- 2.1 Datastores desde particiones predefinidas --------------
        trainDir = fullfile(datasetDir, "train");
        assert(isfolder(trainDir), ...
            "No existe la partición obligatoria de entrenamiento: %s", trainDir);

        imdsTrain = makeImds(trainDir);
        classNames = string(categories(imdsTrain.Labels));
        numClasses = numel(classNames);

        fprintf("Partición train: %d imágenes, %d clases: %s\n", ...
            numel(imdsTrain.Files), numClasses, strjoin(classNames, ", "));
        disp(countEachLabel(imdsTrain));  % revisar balance de clases

        % Validación: carpeta "valid" (Roboflow) o "val"; si no, split.
        valDir = firstExistingFolder(datasetDir, ["valid", "val"]);
        if strlength(valDir) > 0
            imdsVal = alignClasses(makeImds(valDir), classNames, "valid");
        else
            warning(['No se encontró ''valid'' ni ''val''; se separa un ' ...
                '%.0f %% de train como validación.'], 100 * autoValFraction);
            [imdsTrain, imdsVal] = splitEachLabel(imdsTrain, ...
                1 - autoValFraction, "randomized");
        end

        % Test: opcional; si falta, se reutiliza validación (con aviso).
        testDir = firstExistingFolder(datasetDir, "test");
        if strlength(testDir) > 0
            imdsTest = alignClasses(makeImds(testDir), classNames, "test");
            testSplitName = "test";
        else
            warning(['No se encontró ''test''; las métricas finales se ' ...
                'calcularán sobre la validación (interpretarlas con cautela).']);
            imdsTest = copy(imdsVal);
            testSplitName = "valid (sin test)";
        end

        % ---- 2.2 Red base adaptada al nº de clases -----------------------
        % Con "NumClasses", imagePretrainedNetwork reemplaza la capa final
        % de clasificación automáticamente.
        net = imagePretrainedNetwork(baseModel, "NumClasses", numClasses);
        inputSize = net.Layers(1).InputSize;
        fprintf("Red base '%s' | entrada [%d %d %d]\n", baseModel, inputSize);

        % ---- 2.3 Aumento de datos y redimensionado -----------------------
        % El aumento se aplica SOLO a train; val/test solo redimensionan.
        augmenter = imageDataAugmenter( ...
            "RandXReflection",  true, ...
            "RandXTranslation", [-15 15], ...
            "RandYTranslation", [-15 15], ...
            "RandScale",        [0.9 1.1]);

        augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
            "DataAugmentation", augmenter, "ColorPreprocessing", "gray2rgb");
        augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
            "ColorPreprocessing", "gray2rgb");
        augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest, ...
            "ColorPreprocessing", "gray2rgb");

        % ---- 2.4 Entrenamiento -------------------------------------------
        options = trainingOptions("adam", ...
            "InitialLearnRate",     initialLearnRate, ...
            "MaxEpochs",            maxEpochs, ...
            "MiniBatchSize",        miniBatchSize, ...
            "Shuffle",              "every-epoch", ...
            "ValidationData",       augVal, ...
            "ValidationFrequency",  max(1, floor(numel(imdsTrain.Files)/miniBatchSize)), ...
            "Metrics",              "accuracy", ...
            "ExecutionEnvironment", "auto", ...     % GPU si está disponible
            "Plots",                "training-progress", ...
            "Verbose",              true);

        % "crossentropy" espera probabilidades: las redes de
        % imagePretrainedNetwork ya incluyen softmax al final.
        [net, info] = trainnet(augTrain, net, "crossentropy", options);

        % ---- 2.5 Evaluación final -----------------------------------------
        scoresTest = minibatchpredict(net, augTest, ...
            "MiniBatchSize", miniBatchSize);
        predTest = scores2label(scoresTest, classNames);
        trueTest = imdsTest.Labels;

        testAccuracy = mean(predTest(:) == trueTest(:));
        fprintf("\nPrecisión en '%s': %.2f %%\n", ...
            testSplitName, 100 * testAccuracy);

        figCM = figure("Name", "Matriz de confusión");
        confusionchart(trueTest, predTest, ...
            "RowSummary", "row-normalized", "Title", ...
            sprintf("%s | Precisión global: %.1f %%", ...
            testSplitName, 100 * testAccuracy));
        exportgraphics(figCM, "confusion_test.png", "Resolution", 150);

    otherwise
        error("Modo no válido: '%s'. Use ""train"" o ""export"".", mode);
end

%% 3. Guardado (contrato común a ambos modos) =============================
save(outputFile, "net", "classNames", "inputSize", "baseModel", ...
    "mode", "info", "testAccuracy");

fprintf("\nModelo guardado en '%s' (modo '%s').\n", outputFile, mode);
fprintf("Siguiente paso: testClassifier.m (prueba local) y luego\n");
fprintf("deployClassifierJetson.m (despliegue en la Jetson).\n");

%% ========================= Funciones locales ============================

function imds = makeImds(folder)
%MAKEIMDS Crea un imageDatastore etiquetado por nombre de subcarpeta.
imds = imageDatastore(folder, ...
    "IncludeSubfolders", true, ...
    "LabelSource",       "foldernames");
end

function folderOut = firstExistingFolder(rootDir, names)
%FIRSTEXISTINGFOLDER Devuelve la primera subcarpeta existente de la lista
%   (p. ej. ["valid","val"]) o "" si ninguna existe.
folderOut = "";
for n = names(:).'
    candidate = fullfile(rootDir, n);
    if isfolder(candidate)
        folderOut = candidate;
        return
    end
end
end

function imds = alignClasses(imds, classNames, splitName)
%ALIGNCLASSES Alinea las clases de una partición con las de entrenamiento.
%   Avisa si la partición contiene clases desconocidas (imposibles de
%   predecir) o si le faltan clases, y unifica el conjunto/orden de
%   categorías para que las comparaciones y scores2label sean coherentes.
found   = string(categories(imds.Labels));
extra   = setdiff(found, classNames);
missing = setdiff(classNames, found);

if ~isempty(extra)
    warning(['La partición ''%s'' contiene clases ausentes en train ' ...
        '(%s): la red nunca podrá predecirlas.'], ...
        splitName, strjoin(extra, ", "));
end
if ~isempty(missing)
    warning("A la partición '%s' le faltan clases de train (%s).", ...
        splitName, strjoin(missing, ", "));
end

imds.Labels = setcats(imds.Labels, cellstr(classNames));
end
