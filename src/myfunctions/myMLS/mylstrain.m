function ClassifierLS = mylstrain(data,labels,varargin)
%MYLSTRAIN - least squares classifier training.
%   To train the least squares classifiers, namely X in the Ax=b equation 
%   or AX=B, which means their are more than one training tasks.
%  
%   ClassifierLS = mylstrain(data,labels)
%   ClassifierLS = mylstrain(data,labels,dimension)
%   ClassifierLS = mylstrain(data,labels,dimension,disp)
%   ClassifierLS = mylstrain(data,labels,dimension,disp,interval_num)
% 
%   Input - 
%   data:       a m*n matrix A in the equation AX=B where m is sampling quantity;
%   labels:     a m*1 vector with entries inside representing the category;
%   dimension:  the parameter indicates organization of parameter "data":
%               dimension=1: sampling quantity == size(data,1)
%               dimension=2: sampling quantity == size(data,2)
%   disp:       a Boolean variable, deciding whether to show the figures 
%               of Boolean classifier distribution or not.
%   Output - 
%   ClassifierLS:   a struct containing information about the trained LS
%                   classifier;
%   .X:             the n*k matrix representing the aggregate of k weighting 
%                   vectors from k different tasks, and X = [x1,x2,...xk];
%   .feedback_labels:   predicted labels of the training-set itself using 
%                       the LS classifier trained.
%   .feedback_cfsmtx:   the confusion matrix of feedback_labels and raw
%                       input labels.
% 
%   Copyright (c) 2019 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% parameters check
% 参数数量检测
narginchk(2,5);
nargoutchk(1,1);
% 获取剩余输入参数(varargin)数目
narg = numel(varargin);
% 预先定义参数
dimension = 1;      % row: sampling quantity 
disp = false;
interval_num = 50;
% 获取剩余输入参数(varargin)值
switch narg
    case 0
    case 1
        dimension = varargin{:};      
    case 2
        [dimension,disp] = varargin{:};
    case 3
        [dimension,disp,interval_num] = varargin{:};
    otherwise
        error('Error! Number of input parameters erong.');
end
% 参数检查
if dimension~=1
    data = data';
end
[mum_data,n_fts] = size(data);
if mum_data<n_fts
    error('Error! The input matrix A should be tall, or at least be square.');
end
if interval_num <= 0
    error('Error! Parameter "interval_num" should be a positive number.');
end
% 准备用于训练的矩阵 B:
% the m*k matrix representing the aggregate of observation outputs from
% k different tasks, and B = [b1,b2,...bk].
classes = size(mynumstatistic(labels),1);
B = -1*ones(mum_data,classes);
for i=1:classes
    for j = 1:mum_data
        if labels(j)==i
            B(j,i)=1;
        end
    end
end
if ~ismatrix(data)||~ismatrix(B)||size(data,1)~=size(B,1)
    error('Error! Input parameters "data" and "labels" error.');
end
%% get least squares classifiers
X = zeros(n_fts,classes);
% least squares formula: x = inv((A')*A)*(A')*b.
for i=1:classes
    X(:,i) = (data'*data)\data'*B(:,i);
end
%% feedback correction rate on the training-set itself
feedback_labels = mylsclassify(data,X);
feedback_cfsmtx = mycfsmtx(labels,feedback_labels);     % 训练集混淆矩阵
%% draw the Boolean classifier distribution
B_trained = data*X;
if disp
    for i=1:classes
        b_trained = B_trained(:,i);
        Idx_start = find(B(:,i)==1,1);
        Idx_end = find(B(:,i)==1,1,'last');
        pos_num = Idx_end-Idx_start+1;
        neg_num = length(b_trained)-pos_num;
        [a,c1] = hist(b_trained(Idx_start:Idx_end),interval_num);    % positive
        b_trained(Idx_start:Idx_end) = [];
        [b,c2] = hist(b_trained,interval_num);                       % negative
        figure;bar(c1,a/pos_num,'b');
        hold on;
        bar(c2,b/neg_num,'r');
        ylabel('Fraction');legend('Positive','negative');
        title(['Boolean classifier distribution for category',num2str(i),' VS others']);
    end
end
%% parameters output
ClassifierLS.X = X;
ClassifierLS.feedback_labels = feedback_labels;
ClassifierLS.feedback_cfsmtx = feedback_cfsmtx;
end