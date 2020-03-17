function vectcode = myvectorquantization(codebook,fts)

% code = myvectquantization([800*64],[116*64])
cb_len = size(codebook,1);
ft_len = size(fts,1);
if size(codebook,2) ~= size(fts,2)
    error('Error! Input parameter error.');
end
vectcode = zeros(cb_len,1);
for i=1:ft_len      % ¶Ô116¸ö 64-vector
    code_single = myvq_single(codebook,fts(i,:));
    vectcode = vectcode+code_single;
end
vectcode = vectcode';
end