function vallabel_predict = myNaiveBayesValidation(TrainingSets,ValidationSets)
% Validation: validate model achieved by testing sets with validation sets.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

ftsnum = size(TrainingSets(1).feature,2);
for i=1:2
    for j=1:ValidationSets(i).number              % for each voice
        data = ValidationSets(i).feature(j,:);
        for k=1:ftsnum
            % probability of being female voice
            ValidationSets(i).results(j,1)=...
                TrainingSets(1).feature_prob(data(k),k)*ValidationSets(i).results(j,1);
            % probability of being male voice
            ValidationSets(i).results(j,2)=...
                TrainingSets(2).feature_prob(data(k),k)*ValidationSets(i).results(j,2);
        end
        if ValidationSets(i).results(j,1) > ValidationSets(i).results(j,2)
            % this is female voice
            ValidationSets(i).results(j,3) = 1;
        else
            % this is male voice
            ValidationSets(i).results(j,3) = 2;
        end
    end
end

%% return the result
vallabel_predict = [ValidationSets(1).results(:,3);ValidationSets(2).results(:,3)];

end