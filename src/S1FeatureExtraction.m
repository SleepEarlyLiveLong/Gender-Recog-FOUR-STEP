% Name:     S1FeatureExtraction.m
% Function: Extract features of a certain voice dataset.

% Copyright (c) 2019 CHEN Tianyang
% more info contact: tychen@whu.edu.cn

%% 准备工作
clear;close all;
addpath(genpath([pwd,'\myfunctions']));
addpath(genpath([pwd,'\libsvm322']));

%% 提取特征并存储 - 结构体
predir = uigetdir();
fold_list = dir(predir);
category_num = length(fold_list)-2;
% 获得信息构建结构体
VOICES(category_num,1) = struct('name',[],'data',[],'num',[]);
% 读取图像到结构体中
fts_num = 13;       % 提取的特征数目(可选数字: 1-39)
for i=1:category_num
    VOICES(i).name = fold_list(i+2).name;
    % 每个类中的若干个文件
    file_list = dir([predir,'\',VOICES(i).name]);
%     VOICES(i).num = min(length(file_list)-2,50+round(10*rand()));   % 随机抽取
    VOICES(i).num = length(file_list)-2;                            % 所有数据
    VOICES(i).data = zeros(VOICES(i).num,fts_num);
    for j=1:VOICES(i).num
        [currentvoice,fs] = audioread([predir,'\',VOICES(i).name,'\',file_list(j+2).name]);
        feature = myfeature(currentvoice,fs);       % 提取各种特征,得到行向量
        VOICES(i).data(j,:) = feature(1:fts_num);               % 存入结构体中
    end
    fprintf('第%d类/共%d类语音(共%d条数据)特征提取完毕\n',i,category_num,VOICES(i).num);
end
save('database\struct_freqfts','VOICES');
fprintf('特征已存储\n');
% figure;
% h1 = histogram(VOICES(1).data);hold on;
% h2 = histogram(VOICES(2).data);
% h1.FaceColor = 'r';h1.Normalization = 'probability';h1.BinWidth = 1;
% h2.FaceColor = 'b';h2.Normalization = 'probability';h2.BinWidth = 1;
% legend('female','male');
% xlabel('pitch/Hz');ylabel('Probability');title('Pitch Distribution for male and female ');

%% 扫尾
rmpath([pwd,'\myfunctions']);
rmpath(genpath([pwd,'\libsvm322']));