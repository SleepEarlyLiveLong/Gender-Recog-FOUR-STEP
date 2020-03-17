function feature_qualify = mydiscretization2(feature_max,feature_min,ftsRange,feature)
%MYDISCRETIZATION2 - The data discreization function (WITH KEY PARAMETERS SPECIFIED).
%   
%   feature_qualify = mydiscretization2(feature_max,feature_min,ftsRange,feature)
% 
%   Input - 
%   feature_max: max value of the input feature;
%   feature_min: min value of the input feature;
%   ftsRange:    step number, which defines the discretization accuracy;
%   feature:     the input feature array.

%   Output - 
%   feature_qualify: the output feature array after discreization.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 
if ~isvector(feature_max)||~isvector(feature_min)||~isvector(feature)||ftsRange<=0
    error('Input parameter error!');
end
feature_max = feature_max(:)';
feature_min = feature_min(:)';
feature = feature(:)';
if length(feature_max)~=length(feature_min)||length(feature_max)~=length(feature)
    error('Input parameter error!');
end
ftsnum = length(feature);
feature_qualify = zeros(1,ftsnum);
for i = 1:ftsnum
    pdiff = (feature_max(i)-feature_min(i))/ftsRange;
    for level = 0:ftsRange-1
        if feature(i) < feature_min(i)
            feature_qualify(i) = 1;
            break;
        elseif feature(i) >= (feature_min(i)+level*pdiff) && feature(i) < (feature_min(i)+(level+1)*pdiff)
            feature_qualify(i) = level+1;
            break;
        elseif feature(i) >= feature_max(i)
            feature_qualify(i) = ftsRange;
        end
    end
end      
end  