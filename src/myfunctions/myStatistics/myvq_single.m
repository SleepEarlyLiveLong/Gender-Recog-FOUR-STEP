function code_single = myvq_single(codebook,fts_single)

% code = myvectquantization([800*64],[1*64])
cb_len = size(codebook,1);
code_single = zeros(cb_len,1);
dist = zeros(cb_len,1);
for i=1:cb_len
    dist(i) = norm(codebook(i,:)-fts_single);
end
[~,seq] = max(dist);
code_single(seq) = 1;
end