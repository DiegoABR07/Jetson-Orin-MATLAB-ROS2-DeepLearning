function data = augmentYoloData(data)
%AUGMENTYOLODATA Aumento de datos en línea para entrenamiento de detectores.
%
%   data = augmentYoloData(data)
%
%   Pensada para usarse con TRANSFORM sobre el datastore de entrenamiento:
%       dsTrainAug = transform(dsTrain, @augmentYoloData);
%
%   Transformaciones por muestra {imagen, cajas, etiquetas}:
%       1. Conversión a RGB si la imagen es en escala de grises.
%       2. Jitter de color en espacio HSV (tono, saturación, brillo).
%       3. Volteo horizontal aleatorio + escalado aleatorio (zoom),
%          transformando también las cajas con BBOXWARP.
%
%   Si tras la transformación geométrica no sobrevive ninguna caja
%   (solapamiento < 25 %), se conserva la muestra original para no
%   entrenar con imágenes vacías.
%
%   Ver también: ensureRGBData, bboxwarp, jitterColorHSV

data = ensureRGBData(data);

for ii = 1:size(data, 1)
    I      = data{ii, 1};
    bboxes = data{ii, 2};
    labels = data{ii, 3};
    sz     = size(I);

    % --- 1) Jitter fotométrico ------------------------------------------
    I = jitterColorHSV(I, ...
        "Contrast",   0.2, ...
        "Hue",        0.05, ...
        "Saturation", 0.2, ...
        "Brightness", 0.2);

    % --- 2) Transformación geométrica aleatoria --------------------------
    tform = randomAffine2d("XReflection", true, "Scale", [1.0, 1.15]);
    rout  = affineOutputView(sz, tform, "BoundsStyle", "centerOutput");
    Iw    = imwarp(I, tform, "OutputView", rout);

    [bboxesW, indices] = bboxwarp(bboxes, tform, rout, ...
        "OverlapThreshold", 0.25);

    if isempty(indices)
        data(ii, :) = {I, bboxes, labels};
    else
        data(ii, :) = {Iw, bboxesW, labels(indices)};
    end
end
end
