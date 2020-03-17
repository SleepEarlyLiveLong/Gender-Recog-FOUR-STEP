% Name:     S3GenderRecogTrainVal.m
% Function: Train and validate models based on the aforementioned training and validation sets.

% Copyright (c) 2019 CHEN Tianyang
% more info contact: tychen@whu.edu.cn

%% 准备工作
clear;close all;
addpath(genpath([pwd,'\myfunctions']));
addpath(genpath([pwd,'\libsvm322']));
predir = 'D:\My Matlab Files\GenderRecog';
load([predir,'\model\train_freqfts.mat']);
load([predir,'\model\test_freqfts.mat']);

%% 分类
%% ---------------------------- Naive Bayes ------------------------
% 预处理-量化
ftsRange = 20;
ftsnum = size(traindata,2);
traindata_qualify = zeros(size(traindata));
valdata_qualify = zeros(size(valdata));
for i=1:ftsnum
    traindata_qualify(:,i) = mydiscretization(traindata(:,i),ftsRange);
    valdata_qualify(:,i) = mydiscretization(valdata(:,i),ftsRange);
end
% 中间结果记录 量化分析
% figure;k=1;for i=1:13 subplot(5,6,k);plot(traindata(:,i));subplot(5,6,k+1);plot(traindata_qualify(:,i));k=k+2;end

% Naive Bayes 训练 and 测试
[TrainingSets,ValidationSets] = myNaiveBayesTrain(traindata_qualify,trainlabel,...
    valdata_qualify,vallabel,ftsRange);
predicted_label = myNaiveBayesValidation(TrainingSets,ValidationSets);
cfsmax = mycfsmtx(vallabel,predicted_label);

%% ----------------------------- SVM -------------------------------
% Rough optimization 粗寻优
[cmin,cmax,gmin,gmax,v,cstep,gstep,accstep] = deal(-15,15,-15,15,3,2,2,2);
[acc_temp,c_temp,g_temp] = SVMcgForClass(trainlabel,traindata,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
% Refined optimization 精寻优
[cmin,cmax,gmin,gmax,v,cstep,gstep,accstep] = deal(log(c_temp)/log(2)-2,log(c_temp)/log(2)+2,...
    log(g_temp)/log(2)-2,log(g_temp)/log(2)+2,3,0.25,0.25,0.5);
[bestacc,bestc,bestg] = SVMcgForClass(trainlabel,traindata,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
% 全数据训练
cmd = [' -s ',num2str(0),' -c ',num2str(bestc),' -g ',num2str(bestg)];
svm_model = svmtrain(trainlabel,traindata,cmd);
% 测试
[predicted_label2, accuracy, decision_values] = svmpredict(vallabel,valdata,svm_model);
cfsmtxs2 = mycfsmtx(vallabel,predicted_label2);

%% ---------------------- Distance Determination -------------------
Result = myDistDetermine(traindata,trainlabel,valdata);
predicted_label3 = Result.predicted_label;
cfsmtxs3 = mycfsmtx(vallabel,predicted_label3);

%% ------------------------------- KNN -----------------------------
% experiment proves that 7-NN performs best
knn_model = fitcknn(traindata,trainlabel,'NumNeighbors',7); 
predicted_label4 = knn_model.predict(valdata);
cfsmtxs4 = mycfsmtx(vallabel,predicted_label4);

%% -------------------------- kmeans+KNN ---------------------------
% kmeans
clusters = 100;
train_class_num = mynumstatistic(trainlabel);
train_female_num = train_class_num(1,2);
train_male_num = train_class_num(2,2);
[~,traindata_new_f,~,~,~] = mykmeans( traindata(1:train_female_num,:),clusters );
[~,traindata_new_m,~,~,~] = mykmeans( traindata(train_female_num+1:length(trainlabel),:),clusters );
traindata_new = [traindata_new_f;traindata_new_m];
trainlabel_new = [ones(clusters,1);2*ones(clusters,1)];
% % KNN
% Ktry = 100;
% Accr = zeros(Ktry,1);
% for K = 1:Ktry
    knn_model = fitcknn(traindata_new,trainlabel_new,'NumNeighbors',3); % K选值
    predicted_label5 = knn_model.predict(valdata);
    cfsmtxs5 = mycfsmtx(vallabel,predicted_label5);
%     Accr(K) = cfsmtxs5(end,end);
% end
% % 中间结果记录
% figure;plot(Accr);xlabel('K');ylabel('accuracy rate');title('100means + KNN');

%% ------------------------------ MLS ------------------------------
ClassifierLS = mylstrain(traindata,trainlabel,1,'true');
predicted_label6 = mylsclassify(valdata,ClassifierLS.X);
cfsmtx6 = mycfsmtx(vallabel,predicted_label6);

%% 扫尾工作
rmpath([pwd,'\function']);
rmpath(genpath([pwd,'\libsvm322']));