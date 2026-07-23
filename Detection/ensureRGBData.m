function data = ensureRGBData(data)
%ENSURERGBDATA Garantiza que las imágenes del datastore tengan 3 canales.
%
%   data = ensureRGBData(data)
%
%   El detector base (preentrenado en COCO) espera entradas RGB. Esta
%   función normaliza cada muestra {imagen, cajas, etiquetas} a 3 canales.
%
%   Uso típico (validación, sin aumento de datos):
%       dsVal = transform(dsVal, @ensureRGBData);
%
%   Ver también: augmentYoloData

for ii = 1:size(data, 1)
    I = data{ii, 1};
    if size(I, 3) == 1
        I = repmat(I, [1, 1, 3]);       % gris -> RGB
    elseif size(I, 3) > 3
        I = I(:, :, 1:3);               % descartar canal alfa
    end
    data{ii, 1} = I;
end
end
