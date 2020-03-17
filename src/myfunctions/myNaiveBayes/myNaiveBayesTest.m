function [predicted_label,female_prob,male_prob] = myNaiveBayesTest(TrainingSets,featuretest)
% myNaiveBayesTest: test a new data with the model achieved.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

if ~isvector(featuretest)
    error('Input parameter error!');
end
featuretest = featuretest(:)';
if size(TrainingSets(1).feature,2)~=size(featuretest,2)
    error('Input parameter error!');
end
ftsnum = size(featuretest,2);

female_prob = 1;
male_prob = 1;
for k=1:ftsnum
    % probability of being female voice
    female_prob=TrainingSets(1).feature_prob(featuretest(k),k)*female_prob;
    % probability of being male voice
    male_prob=TrainingSets(2).feature_prob(featuretest(k),k)*male_prob;
end
if female_prob > male_prob
    % this is female voice
    predicted_label = 1;
else
    % this is male voice
    predicted_label = 2;
end
end