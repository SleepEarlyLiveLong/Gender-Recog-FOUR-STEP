function predicted_labels = mylsclassify(data,X)
%MYLSCLASSIFER - least squares multi-classifiers.
%   To use the trained LS classifier for JUST ONE MULTI-CLASSIFY task.
%   
%   B_predict = mylsclassify(A,X)
% 
%   Input -
%   data:   a m*n matrix A in the equation AX=B;
%   X:      the n*k matrix representing the aggregate of k weighting vectors
%           trained from fuction 'mylstrain' from the k-CLASSIFY task,
%           and X = [x1,x2,...xk];
%   Output -
%   predicted_labels:	the predicted labels for "data" using "X" offered;       
%   Middle -
%   B_predict;  the m*k matrix representing the aggregate of predicted 
%               outputs for the k-CLASSIFY task, and
%               B_predict = [b_predict1,b_predict2,...b_predictk].
% 
%   Copyright (c) 2019 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% parameters check
if ~ismatrix(data)||~ismatrix(X)||size(data,2)~=size(X,1)
    error('Error! Input parameters error.');
end
[m,~] = size(data);
[~,classes] = size(X);
%% classifying
B_predict = zeros(m,classes);
for i=1:classes
    B_predict(:,i) = data*X(:,i);
end
val = zeros(m,1);
pos = zeros(m,1);
for i=1:m
    [val(i),pos(i)] = max(B_predict(i,:));
end
%% parameters output
predicted_labels = pos;
end