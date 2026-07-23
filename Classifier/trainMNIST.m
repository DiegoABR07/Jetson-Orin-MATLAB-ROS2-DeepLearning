%TRAINMNIST Entrena una CNN sencilla con DigitDataset y la exporta al pipeline.
%
%   Red para MNIST:
%       entrada 28x28x1 -> convolución (10 filtros de 3x3) -> tanh ->
%       capa densa (10 unidades) -> softmax
%   entrenada con el DigitDataset de MATLAB (10 000 imágenes de dígitos
%   0-9, 28x28 en escala de grises, incluidas con Deep Learning Toolbox),
%   particionado en train / validación / test.
%
%   El script también incluye una segunda arquitectura seleccionable
%   (conv + ReLU + max pooling + dropout):
%   basta cambiar el parámetro 'architecture'.
%
%   Salida (MISMO formato que trainClassifier.m, por lo que el resto del
%   pipeline -testClassifier, deployClassifierJetson, viewClassifierROS2-
%   funciona sin cambios):
%       trainedClassifier.mat con: net, classNames, inputSize, baseModel,
%       mode, info, testAccuracy.
%
%   AJUSTES PARA LA WEBCAM (leer antes de desplegar):
%   1) INPUT_SIZE del nodo: en classifierJetsonNode.m cambiar
%          INPUT_SIZE = [28 28 1];
%      y volver a ejecutar deployClassifierJetson.m (lo verifica solo).
%      El nodo ya soporta entrada monocanal: convierte el cuadro rgb8 de
%      la cámara a gris, lo redimensiona a 28x28 y publica esa misma
%      vista (replicada a 3 canales) en /classifier/image_view, de modo
%      que en el visor se observa EXACTAMENTE lo que ve la red (28x28,
%      pixelado al ampliarse: es normal y didáctico).
%   2) Encuadre: los 640x480 completos se comprimen a 28x28, así que el
%      dígito debe LLENAR el cuadro y estar centrado, con fondo liso.
%   3) Polaridad: DigitDataset tiene dígitos CLAROS sobre fondo OSCURO.
%      Un dígito de tinta sobre papel blanco es lo contrario y degrada la
%      precisión. Opciones: mostrar dígitos claros sobre fondo oscuro
%      (p. ej. en una pantalla) o activar includeInvertedCopies = true,
%      que entrena también con las imágenes invertidas para que el modelo
%      tolere ambas polaridades frente a la cámara.
%
%   Ver también: trainClassifier, classifyImage, deployClassifierJetson

clear; clc; close all;
rng(0);                                   % reproducibilidad

%% 1. Parámetros de usuario ================================================
architecture = "conv-tanh";               % "conv-tanh"
                                          % "conv-relu-pool-dropout" (próxima sesión)

trainFraction = 0.70;                     % 7000 imágenes de entrenamiento
valFraction   = 0.15;                     % 1500 validación / 1500 test

includeInvertedCopies = true;            % true: robustez de polaridad p/ webcam

maxEpochs        = 70;
miniBatchSize    = 256;
initialLearnRate = 1e-3;

outputFile = "trainedClassifier.mat";     % OJO: sobrescribe el modelo previo
                                          % del pipeline; renombrar antes el
                                          % anterior si se desea conservar.

%% 2. Dataset: DigitDataset de MATLAB =====================================
digitDatasetPath = fullfile(matlabroot, "toolbox", "nnet", "nndemos", ...
    "nndatasets", "DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource",       "foldernames");

classNames = string(categories(imds.Labels));   % "0" ... "9"
fprintf("DigitDataset: %d imágenes, %d clases.\n", ...
    numel(imds.Files), numel(classNames));

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, ...
    trainFraction, valFraction, "randomized");

% Muestra del dataset (Ojitooo: dígitos claros sobre fondo oscuro).
figure("Name", "Muestras de DigitDataset");
montage(imdsTrain.Files(randperm(numel(imdsTrain.Files), 25)), ...
    "Size", [5 5], "BorderSize", 2);
title("DigitDataset: dígitos claros sobre fondo oscuro (28x28)");

%% 3. Arquitectura de la red ==============================================
inputSize = [28 28 1];

switch architecture
    case "conv-tanh"
        % Red del ejercicio: 1 conv (10 filtros 3x3) + tanh + densa(10) +
        % softmax. "Padding=same" conserva el tamaño espacial 28x28.
        layers = [
            imageInputLayer(inputSize)                     % normaliza (zerocenter)
            convolution2dLayer(3, 10, "Padding", "same")   % 10 filtros de 3x3
            tanhLayer                                      % activación tanh
            fullyConnectedLayer(numel(classNames))         % densa: 10 salidas
            softmaxLayer                                   % probabilidades
        ];

    case "conv-relu-pool-dropout"
        % Variante: conv + ReLU + max pooling +
        % dropout. El pooling reduce 28x28 -> 14x14; el dropout (25 %)
        % regulariza apagando activaciones aleatorias al entrenar.
        layers = [
            imageInputLayer(inputSize)
            convolution2dLayer(3, 16, "Padding", "same")
            reluLayer
            maxPooling2dLayer(2, "Stride", 2)
            dropoutLayer(0.25)
            fullyConnectedLayer(numel(classNames))
            softmaxLayer
        ];

    otherwise
        error("Arquitectura no reconocida: %s", architecture);
end

net = dlnetwork(layers);

%% 4. Datos en memoria (el dataset es pequeño: ~8 MB) =====================
% Cargar a arreglos permite el aumento opcional de polaridad y acelera el
% entrenamiento (sin lecturas de disco por iteración).
[XTrain, TTrain] = loadDigitImages(imdsTrain, inputSize);
[XVal,   TVal]   = loadDigitImages(imdsVal,   inputSize);
[XTest,  TTest]  = loadDigitImages(imdsTest,  inputSize);

if includeInvertedCopies
    % Copias con polaridad invertida (dígito oscuro sobre fondo claro),
    % como se vería tinta sobre papel blanco en la webcam.
    XTrain = cat(4, XTrain, 255 - XTrain);
    TTrain = [TTrain; TTrain];
    fprintf("Aumento de polaridad activado: %d muestras de entrenamiento.\n", ...
        numel(TTrain));
end

%% 5. Entrenamiento ========================================================
options = trainingOptions("adam", ...
    "InitialLearnRate",     initialLearnRate, ...
    "MaxEpochs",            maxEpochs, ...
    "MiniBatchSize",        miniBatchSize, ...
    "Shuffle",              "every-epoch", ...
    "ValidationData",       {XVal, TVal}, ...
    "ValidationFrequency",  max(1, floor(numel(TTrain)/miniBatchSize)), ...
    "Metrics",              "accuracy", ...
    "ExecutionEnvironment", "auto", ...
    "Plots",                "training-progress", ...
    "Verbose",              true);

% "crossentropy" espera probabilidades: la red termina en softmax.
[net, info] = trainnet(XTrain, TTrain, net, "crossentropy", options);

%% 6. Análisis del modelo ==================================================
fprintf("\n===== Análisis del modelo ('%s') =====\n", architecture);
disp(net.Layers);                 % capas y tamaños
summary(net);                     % nº de parámetros aprendibles por capa

% Inspector interactivo de la red (grafico, activaciones, parámetros).
% Comentar esta línea si se ejecuta sin entorno gráfico.
analyzeNetwork(net);

%% 7. Evaluación en test ===================================================
scoresTest = minibatchpredict(net, XTest, "MiniBatchSize", miniBatchSize);
predTest   = scores2label(scoresTest, classNames);

testAccuracy = mean(predTest(:) == TTest(:));
fprintf("Precisión en test: %.2f %%\n", 100 * testAccuracy);

figCM = figure("Name", "Matriz de confusión (test)");
confusionchart(TTest, predTest, ...
    "RowSummary", "row-normalized", "Title", ...
    sprintf("DigitDataset test | %s | Precisión: %.1f %%", ...
    architecture, 100 * testAccuracy));
exportgraphics(figCM, "mnist_confusion_test.png", "Resolution", 150);

% ---- Resultados cualitativos: 16 dígitos de test con su predicción -----
figQ = figure("Name", "Predicciones de ejemplo (test)");
tl = tiledlayout(4, 4, "TileSpacing", "compact");
idx = randperm(size(XTest, 4), 16);
for k = idx
    nexttile;
    imshow(uint8(XTest(:, :, 1, k)), "InitialMagnification", "fit");
    ok = predTest(k) == TTest(k);
    if ok, c = [0 0.5 0]; else, c = [0.8 0 0]; end
    title(sprintf("pred %s | real %s", string(predTest(k)), ...
        string(TTest(k))), "Color", c, "FontSize", 9);
end
title(tl, "Verde = acierto, rojo = error");
exportgraphics(figQ, "mnist_predicciones_test.png", "Resolution", 150);

%% 8. Exportación con el contrato del pipeline ============================
baseModel = "mnist-" + architecture;
mode      = "train";

save(outputFile, "net", "classNames", "inputSize", "baseModel", ...
    "mode", "info", "testAccuracy");

fprintf("\nModelo guardado en '%s' (contrato del pipeline).\n\n", outputFile);
fprintf("=========== SIGUIENTES PASOS PARA LA JETSON ===========\n");
fprintf("1) En classifierJetsonNode.m cambiar:  INPUT_SIZE = [28 28 1];\n");
fprintf("2) Ejecutar deployClassifierJetson.m (verifica INPUT_SIZE solo).\n");
fprintf("3) Webcam: el dígito debe LLENAR el cuadro, centrado, fondo liso.\n");
fprintf("   DigitDataset usa dígitos CLAROS sobre fondo OSCURO: mostrar\n");
fprintf("   dígitos con esa polaridad (p. ej. en una pantalla) o reentrenar\n");
fprintf("   con includeInvertedCopies = true para tolerar tinta en papel.\n");
fprintf("4) La vista en viewClassifierROS2.m será de 28x28 (pixelada al\n");
fprintf("   ampliarse): es exactamente lo que ve la red.\n");

%% ========================= Funciones locales ============================

function [X, T] = loadDigitImages(imds, inputSize)
%LOADDIGITIMAGES Carga un imageDatastore de dígitos a arreglos en memoria.
%   X: single [28 x 28 x 1 x N] en rango 0-255 (la normalización la
%      aplica la capa de entrada de la red, igual que en el resto del
%      pipeline). T: etiquetas categóricas N x 1.
n = numel(imds.Files);
X = zeros([inputSize, n], "single");
for k = 1:n
    I = imread(imds.Files{k});
    if size(I, 3) > 1
        I = rgb2gray(I);                 % defensivo: deben ser grises
    end
    if ~isequal(size(I), inputSize(1:2))
        I = imresize(I, inputSize(1:2)); % defensivo: deben ser 28x28
    end
    X(:, :, 1, k) = single(I);
end
T = imds.Labels(:);
end
