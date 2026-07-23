function [boxes, classIds] = readYoloLabels(txtFile, imageSize)
%READYOLOLABELS Lee un archivo de anotación YOLO (TXT) y lo convierte a píxeles.
%
%   [boxes, classIds] = readYoloLabels(txtFile, imageSize)
%
%   Formato de entrada (una detección por línea, valores normalizados 0-1):
%       <id_clase> <x_centro> <y_centro> <ancho> <alto>
%
%   Entradas:
%       txtFile   - Ruta al .txt de anotación. Si no existe o está vacío,
%                   se devuelven arreglos vacíos (imagen sin objetos).
%       imageSize - Tamaño de la imagen [alto ancho] o [alto ancho canales].
%
%   Salidas:
%       boxes    - M x 4 en formato MATLAB [x y w h] (píxeles, esquina
%                  superior izquierda, 1-based), recortada a la imagen.
%       classIds - M x 1 con los IDs de clase 0-based del TXT.
%
%   Ver también: buildYoloDatastore

arguments
    txtFile   (1,1) string
    imageSize (1,:) double {mustBePositive}
end

H = imageSize(1);
W = imageSize(2);

boxes    = zeros(0, 4);
classIds = zeros(0, 1);

if ~isfile(txtFile)
    return
end

raw = readmatrix(txtFile, "FileType", "text");
if isempty(raw)
    return
end
if size(raw, 2) < 5
    error("readYoloLabels:badFormat", ...
        "El archivo %s no tiene el formato esperado (5 columnas).", txtFile);
end
raw = raw(:, 1:5);                      % ignorar columnas extra
raw = raw(all(isfinite(raw), 2), :);    % descartar filas incompletas

classIds = raw(:, 1);

% Centro normalizado -> esquina en píxeles, recortando a la imagen.
cx = raw(:, 2) * W;   cy = raw(:, 3) * H;
bw = raw(:, 4) * W;   bh = raw(:, 5) * H;

x1 = max(cx - bw/2, 1);
y1 = max(cy - bh/2, 1);
x2 = min(cx + bw/2, W);
y2 = min(cy + bh/2, H);

boxes = [x1, y1, x2 - x1, y2 - y1];

% Filtrar cajas degeneradas (< 1 píxel tras el recorte).
valid    = boxes(:, 3) >= 1 & boxes(:, 4) >= 1;
boxes    = boxes(valid, :);
classIds = classIds(valid);
end
