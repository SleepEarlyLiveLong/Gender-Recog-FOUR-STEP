function [TrainingSets,ValidationSets] = myNaiveBayesTrain(traindata,trainlabel,valdata,vallabel,ftsRange)
% myNaiveBayesTrain: train and get the model we need, then save the data as .mat.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

% construct TrainingSet
ftsnum = size(traindata,2);
train_num = length(trainlabel);
F_train_num = length(find(trainlabel==1));
M_train_num = length(find(trainlabel==2));
val_num = length(vallabel);
F_val_num = length(find(vallabel==1));
M_val_num = length(find(vallabel==2));
TrainingSets = repmat(struct('gender',0,'code',0,'number',0,...
    'feature',zeros(max(M_train_num,F_train_num),ftsnum),...
    'feature_prob',zeros(ftsRange,ftsnum),'gender_prob',0),2,1);

%% 构建专用于 Naive Bayes 的训练集/验证集结构体
% fill the TrainingSet
TrainingSets(1).gender ='female';
TrainingSets(2).gender ='male';
TrainingSets(1).code = 1;
TrainingSets(2).code = 2;
TrainingSets(1).number = F_train_num;
TrainingSets(2).number = M_train_num;
TrainingSets(1).feature = traindata(1:F_train_num,:);
TrainingSets(2).feature = traindata(F_train_num+1:F_train_num+M_train_num,:);
TrainingSets(1).gender_prob = F_train_num/train_num;
TrainingSets(2).gender_prob = M_train_num/train_num;

% fill the ValidationSets
ValidationSets(1).gender ='female';
ValidationSets(2).gender ='male';
ValidationSets(1).code = 1;
ValidationSets(2).code = 2;
ValidationSets(1).number = F_val_num;
ValidationSets(2).number = M_val_num;
ValidationSets(1).feature = valdata(1:F_val_num,:);
ValidationSets(2).feature = valdata(F_val_num+1:F_val_num+M_val_num,:);
ValidationSets(1).gender_prob = F_val_num/val_num;
ValidationSets(2).gender_prob = M_val_num/val_num;
ValidationSets(1).results = ones(ValidationSets(1).number,3);
ValidationSets(2).results = ones(ValidationSets(2).number,3);

%% 
% Calculate conditional probabilities
% 注意:这里在计算时分子分母都加1是为了避免出现概率为0的情况。一旦为0，概率连乘也为0，Bayes方法失效
% 训练集有792条数据，每一个量化阶数字出现的的最小概率是1/(792+1) = 0.0013
for j=1:ftsnum
    for i=1:ftsRange
        TrainingSets(1).feature_prob(i,j) = ...
            (myhowmany(i,TrainingSets(1).feature(:,j))+1)/(M_train_num+1);
        TrainingSets(2).feature_prob(i,j) = ...
            (myhowmany(i,TrainingSets(2).feature(:,j))+1)/(F_train_num+1);
    end
end

end