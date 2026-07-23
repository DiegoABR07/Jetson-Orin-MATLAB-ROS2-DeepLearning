function cfg = loadYoloDataConfig(yamlFile)
%LOADYOLODATACONFIG Lee un archivo data.yaml (estilo Ultralytics) y lo valida.
%
%   cfg = loadYoloDataConfig(yamlFile)
%
%   Entrada:
%       yamlFile - Ruta al archivo YAML de configuración del dataset.
%
%   Salida (struct cfg):
%       .root   - Carpeta raíz del dataset (ruta absoluta resuelta).
%       .train  - Subcarpeta de imágenes de entrenamiento (relativa a root).
%       .val    - Subcarpeta de imágenes de validación.
%       .test   - Subcarpeta de imágenes de prueba ("" si no existe).
%       .nc     - Número de clases.
%       .names  - Nombres de clases (string array, ordenado por ID).
%
%   Notas:
%       * MATLAB no incluye un parser YAML nativo; esta función implementa
%         un parser mínimo suficiente para el data.yaml de Ultralytics
%         (claves simples, listas en línea, listas con "-" y mapas
%         "indice: nombre" indentados). Bloques extra (p. ej. "roboflow:")
%         se ignoran sin error.
%       * Si el export de Roboflow usa la carpeta "valid", declarar en el
%         YAML: val: valid
%
%   Ver también: buildYoloDatastore, trainDetector

arguments
    yamlFile (1,1) string {mustBeFile}
end

lines = readlines(yamlFile);
for i = 1:numel(lines)
    lines(i) = eraseComment(lines(i));
end

raw = struct();
i = 1;
while i <= numel(lines)
    line = lines(i);
    if strlength(strtrim(line)) == 0
        i = i + 1;
        continue
    end
    % Solo claves de nivel superior (sin indentación).
    if ~startsWith(line, " ") && ~startsWith(line, sprintf("\t"))
        tok = regexp(line, "^([A-Za-z0-9_\-]+)\s*:\s*(.*)$", "tokens", "once");
        if isempty(tok)
            i = i + 1;
            continue
        end
        key = tok(1);
        val = strtrim(tok(2));
        if strlength(val) > 0
            raw.(key) = parseScalarOrInlineList(val);
            i = i + 1;
        else
            [blockVal, i] = parseBlock(lines, i + 1);
            raw.(key) = blockVal;
        end
    else
        i = i + 1;
    end
end

% --- Configuración final --------------------------------------------------
cfg = struct();
yamlDir = fileparts(absPath(yamlFile));
if isfield(raw, "path") && strlength(string(raw.path)) > 0
    root = string(raw.path);
    if ~isAbsolutePath(root)
        root = fullfile(yamlDir, root);
    end
else
    root = yamlDir;
end
cfg.root = absPath(root);

cfg.train = getFieldOr(raw, "train", "");
cfg.val   = getFieldOr(raw, "val",   "");
cfg.test  = getFieldOr(raw, "test",  "");

if ~isfield(raw, "names")
    error("loadYoloDataConfig:missingNames", ...
        "El archivo %s no define la clave obligatoria 'names'.", yamlFile);
end
cfg.names = string(raw.names(:));

if isfield(raw, "nc")
    cfg.nc = double(raw.nc);
    if cfg.nc ~= numel(cfg.names)
        error("loadYoloDataConfig:ncMismatch", ...
            "'nc' (%d) no coincide con el número de 'names' (%d).", ...
            cfg.nc, numel(cfg.names));
    end
else
    cfg.nc = numel(cfg.names);
end
end

% ========================= Funciones locales =============================

function line = eraseComment(line)
idx = strfind(char(line), "#");
if ~isempty(idx)
    line = extractBefore(line, idx(1));
end
line = string(deblank(char(line)));
end

function v = parseScalarOrInlineList(val)
val = strtrim(val);
if startsWith(val, "[") && endsWith(val, "]")
    inner = extractBetween(val, 2, strlength(val) - 1);
    parts = strtrim(split(inner, ","));
    v = arrayfun(@stripQuotes, parts);
else
    num = str2double(val);
    if ~isnan(num) && strlength(val) > 0
        v = num;
    else
        v = stripQuotes(val);
    end
end
end

function [v, nextIdx] = parseBlock(lines, startIdx)
items   = strings(0, 1);
indices = [];
i = startIdx;
while i <= numel(lines)
    line = lines(i);
    if strlength(strtrim(line)) == 0
        i = i + 1;
        continue
    end
    if ~startsWith(line, " ") && ~startsWith(line, sprintf("\t")) ...
            && ~startsWith(strtrim(line), "-")
        break
    end
    s = strtrim(line);
    if startsWith(s, "-")
        items(end+1, 1) = stripQuotes(strtrim(extractAfter(s, 1))); %#ok<AGROW>
    else
        tok = regexp(s, "^(\d+)\s*:\s*(.+)$", "tokens", "once");
        if ~isempty(tok)
            indices(end+1, 1) = str2double(tok(1));           %#ok<AGROW>
            items(end+1, 1)   = stripQuotes(strtrim(tok(2))); %#ok<AGROW>
        end
    end
    i = i + 1;
end
if ~isempty(indices)
    [~, order] = sort(indices);
    items = items(order);
end
v = items;
nextIdx = i;
end

function s = stripQuotes(s)
s = strtrim(s);
if (startsWith(s, """") && endsWith(s, """")) || ...
   (startsWith(s, "'")  && endsWith(s, "'"))
    s = extractBetween(s, 2, strlength(s) - 1);
end
s = string(s);
end

function v = getFieldOr(s, f, default)
if isfield(s, f)
    v = string(s.(f));
else
    v = string(default);
end
end

function p = absPath(p)
p = string(p);
if ~isAbsolutePath(p)
    p = fullfile(pwd, p);
end
try
    fo = java.io.File(char(p));
    p  = string(char(fo.getCanonicalPath()));
catch
    % sin Java, se conserva la ruta tal cual
end
end

function tf = isAbsolutePath(p)
p  = char(p);
tf = ~isempty(p) && (p(1) == '/' || p(1) == '\' || ...
     (numel(p) >= 2 && p(2) == ':'));
end
