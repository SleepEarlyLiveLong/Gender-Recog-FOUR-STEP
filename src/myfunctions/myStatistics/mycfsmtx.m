function mtx_cfs = mycfsmtx(label_real,label_predict)
%MYCFSMTX - calculate the confusion matrix.
%   To get the confusion matrix of real output and prediction output.
%   
%   mtx_cfs = mycfsmtx(label_real,label_predict)  
% 
%   Input - 
%   label_real:     a n-vector with real labels inside;
%   label_predict:  a n-vector with predicted labels inside.
%   Output - 
%   mtx_cfs:        the confusion matrix.
% 
%   Copyright (c) 2019 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% parameters test
if ~isvector(label_real)||~isvector(label_predict)||length(label_real)~=length(label_predict)
    error('Error! Two input vectors should be of the same length.');
end
% 预测的标签不应当是实际不存在的标签
if ~all(ismember(label_predict,label_real))
    error('Error! The number of labels should be same.');
end
%% calculate confusion matrix
tabel_real = mynumstatistic(label_real);
w = size(tabel_real,1);
n = length(label_real);
mtx_cfs = zeros(w+1);
for i=1:n
    mtx_cfs(label_real(i),label_predict(i)) = mtx_cfs(label_real(i),label_predict(i))+1;
end
for i=1:w
    mtx_cfs(i,w+1) = mtx_cfs(i,i)/sum(mtx_cfs(i,:));
    mtx_cfs(w+1,i) = mtx_cfs(i,i)/sum(mtx_cfs(:,i));
end
mtx_cfs(w+1,w+1) = sum(trace(mtx_cfs(1:w,1:w)))/sum(sum(mtx_cfs(1:w,1:w)));
end