function Results = myDistDetermine(traindata,trainlabel,valdata)
% 这只是初稿，只能处理2分类的问题，通用性太差，以后找个时间把这个函数扩充下，
% 应当实现对任意有限多分类问题的 Distance Dertermination 方法。

% 对训练集
class_num_train = mynumstatistic(trainlabel);
female_core = mean(traindata(1:class_num_train(1,2),:));        % 是一个行向量
male_core = mean(traindata(class_num_train(1,2)+1:class_num_train(1,2)+class_num_train(2,2),:));    % 是一个行向量

% 逐个判断验证集
valnum = size(valdata,1);
dist = zeros(valnum,2);
predicted_label = zeros(valnum,1);
for i=1:valnum
    dist(i,1) = sum((valdata(i,:)-female_core).^2);
    dist(i,2) = sum((valdata(i,:)-male_core).^2);
    if dist(i,1)>=dist(i,2)
        predicted_label(i) = 2;     % 判断为男性
    else
        predicted_label(i) = 1;     % 判断为女性
    end
end

% 结果输出
DCtraincore = [female_core;male_core];
Results.DCtraincore = DCtraincore;
Results.predicted_label = predicted_label;
end