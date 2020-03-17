% Name:     S2DatasetDivide.m
% Function: Divide the features into training sets and validation sets.

% Copyright (c) 2019 CHEN Tianyang
% more info contact: tychen@whu.edu.cn

%% 准备工作
clear;close all;
addpath(genpath([pwd,'\myfunctions']));
addpath(genpath([pwd,'\libsvm322']));
predir = 'D:\My Matlab Files\GenderRecog';
load([predir,'\database\struct_freqfts.mat']);

%% 划分训练集验证集并存储 - 矩阵(.mat)
category_num = size(VOICES,1);
divide_rate = 0.7;
traindata = [];
trainlabel = [];
valdata = [];
vallabel = [];
for i=1:category_num
    file_num = VOICES(i).num;
    [val_index,train_index] = crossvalind('holdOut',file_num,divide_rate);
    traindata = [traindata;VOICES(i).data(train_index,:)];
    trainlabel = [trainlabel;i*ones(sum(train_index),1)];
    valdata = [valdata;VOICES(i).data(val_index,:)];
    vallabel = [vallabel;i*ones(sum(val_index),1)];
end
save('model\train_freqfts','traindata','trainlabel');
save('model\test_freqfts','valdata','vallabel');
fprintf('训练/验证集已划分完毕\n');  

%% 扫尾工作
rmpath([pwd,'\myfunctions']);
rmpath(genpath([pwd,'\libsvm322']));