function cfsmtx_total = mycfsmtxadd(cfsmtx_in)

if ~iscell(cfsmtx_in)
    error('天阳已经决定了，这个函数只接受元胞数组作为输入，不然你们还是另请高明吧\n');
end
len = length(cfsmtx_in);
side_len = size(cfsmtx_in{1},1);
temp = zeros(side_len);
for i=1:len
    temp = temp+cfsmtx_in{i};
end
cfsmtx_total(1:side_len-1,1:side_len-1) = temp(1:side_len-1,1:side_len-1);

for i=1:side_len-1
    cfsmtx_total(i,side_len) = cfsmtx_total(i,i)/sum(cfsmtx_total(i,:));
    cfsmtx_total(side_len,i) = cfsmtx_total(i,i)/sum(cfsmtx_total(:,i));
end
cfsmtx_total(side_len,side_len) = trace(cfsmtx_total)/sum(sum(cfsmtx_total(1:side_len-1,1:side_len-1)));
end