function [ds, imds, blds, stats] = buildYoloDatastore(cfg, split, opts)
%BUILDYOLODATASTORE Crea los datastores de una partición de un dataset YOLO.
%
%   [ds, imds, blds, stats] = buildYoloDatastore(cfg, split)
%   [...] = buildYoloDatastore(cfg, split, IncludeEmpty=true)
%
%   Convierte un dataset en formato YOLO/Ultralytics (imágenes + TXT) en
%   los objetos que esperan las funciones de entrenamiento y evaluación
%   de Computer Vision Toolbox.
%
%   Entradas:
%       cfg   - Struct devuelto por loadYoloDataConfig.
%       split - "train" | "val" | "test".
%
%   Opciones:
%       IncludeEmpty - (false) Incluir imágenes sin objetos anotados.
%       Verbose      - (true) Mostrar resumen por consola.
%
%   Salidas:
%       ds    - CombinedDatastore {imagen, cajas, etiquetas}, formato de
%               trainYOLOv4ObjectDetector.
%       imds  - imageDatastore de las imágenes incluidas.
%       blds  - boxLabelDatastore ([x y w h] + etiquetas categóricas).
%       stats - Estadísticas (imágenes, cajas por clase, descartes).
%
%   Convención de etiquetas:
%       .../images/... -> .../labels/... con extensión .txt (Ultralytics).
%
%   Ver también: loadYoloDataConfig, readYoloLabels

arguments
    cfg   (1,1) struct
    split (1,1) string {mustBeMember(split, ["train", "val", "test"])}
    opts.IncludeEmpty (1,1) logical = false
    opts.Verbose      (1,1) logical = true
end

if ~isfield(cfg, split) || strlength(cfg.(split)) == 0
    error("buildYoloDatastore:missingSplit", ...
        "La partición '%s' no está definida en el YAML.", split);
end

imgDir = cfg.(split);
if ~(startsWith(imgDir, "/") || startsWith(imgDir, "\") || ...
        (strlength(imgDir) >= 2 && extract(imgDir, 2) == ":"))
    imgDir = fullfile(cfg.root, imgDir);
end
if ~isfolder(imgDir)
    error("buildYoloDatastore:missingFolder", ...
        "No existe la carpeta de imágenes: %s", imgDir);
end

% --- Listar imágenes ------------------------------------------------------
imdsAll = imageDatastore(imgDir, ...
    "IncludeSubfolders", true, ...
    "FileExtensions", [".jpg", ".jpeg", ".png", ".bmp"]);

nAll = numel(imdsAll.Files);
if nAll == 0
    error("buildYoloDatastore:noImages", ...
        "No se encontraron imágenes en %s", imgDir);
end

% --- Leer anotaciones imagen por imagen -----------------------------------
keptFiles  = strings(nAll, 1);
boxesCell  = cell(nAll, 1);
labelsCell = cell(nAll, 1);
nKept      = 0;

nEmpty        = 0;
nMissingLabel = 0;
nBadClass     = 0;
classCounts   = zeros(cfg.nc, 1);

for k = 1:nAll
    imgFile = string(imdsAll.Files{k});
    txtFile = labelPathFor(imgFile);

    info = imfinfo(imgFile);
    [bxs, ids] = readYoloLabels(txtFile, [info.Height, info.Width]);

    if ~isfile(txtFile)
        nMissingLabel = nMissingLabel + 1;
    end

    valid = ids >= 0 & ids <= cfg.nc - 1;
    nBadClass = nBadClass + nnz(~valid);
    bxs = bxs(valid, :);
    ids = ids(valid);

    if isempty(bxs)
        nEmpty = nEmpty + 1;
        if ~opts.IncludeEmpty
            continue
        end
    end

    nKept = nKept + 1;
    keptFiles(nKept)  = imgFile;
    boxesCell{nKept}  = bxs;
    labelsCell{nKept} = categorical(cfg.names(ids + 1), cfg.names);

    if ~isempty(ids)
        classCounts = classCounts + accumarray(ids + 1, 1, [cfg.nc, 1]);
    end
end

keptFiles  = keptFiles(1:nKept);
boxesCell  = boxesCell(1:nKept);
labelsCell = labelsCell(1:nKept);

if nKept == 0
    error("buildYoloDatastore:allEmpty", ...
        "Ninguna imagen de '%s' contiene anotaciones válidas.", split);
end

% --- Construir datastores -------------------------------------------------
imds = imageDatastore(keptFiles);

% NOTA (verificado): para TABLE, el nombre de parámetro debe ser char
% ('VariableNames') o sintaxis de igualdad; un string ("VariableNames")
% se interpreta como variable de datos y provoca "All table variables
% must have the same number of rows".
labelTable = table(boxesCell, labelsCell, ...
    'VariableNames', {'Boxes', 'Labels'});

blds = boxLabelDatastore(labelTable);
ds   = combine(imds, blds);

% --- Estadísticas ----------------------------------------------------------
stats = struct( ...
    "split",              split, ...
    "imageFolder",        string(imgDir), ...
    "numImagesFound",     nAll, ...
    "numImagesUsed",      nKept, ...
    "numImagesEmpty",     nEmpty, ...
    "numMissingLabels",   nMissingLabel, ...
    "numInvalidClassIds", nBadClass, ...
    "classNames",         cfg.names, ...
    "boxesPerClass",      classCounts);

if opts.Verbose
    fprintf("---- Partición '%s' ----\n", split);
    fprintf("  Imágenes encontradas : %d\n", nAll);
    fprintf("  Imágenes utilizadas  : %d\n", nKept);
    fprintf("  Sin objetos          : %d\n", nEmpty);
    fprintf("  Sin archivo TXT      : %d\n", nMissingLabel);
    if nBadClass > 0
        warning("Se descartaron %d cajas con ID de clase fuera de rango.", ...
            nBadClass);
    end
    for c = 1:cfg.nc
        fprintf("  Cajas de '%s': %d\n", cfg.names(c), classCounts(c));
    end
end
end

% ========================= Funciones locales =============================

function txt = labelPathFor(imgFile)
% Convención Ultralytics: .../images/... -> .../labels/... + .txt
[folder, name, ~] = fileparts(imgFile);
sep = string(filesep);
labelFolder = replace(folder, sep + "images", sep + "labels");
if labelFolder == folder
    labelFolder = folder;   % si la carpeta no se llama "images"
end
txt = fullfile(labelFolder, name + ".txt");
end
