function arrayout = mydiscretization(arrayin,varargin)
%MYDISCRETIZATION - The data discreization function.
%   
%   arrayout = mydiscretization(arrayin)
%   arrayout = mydiscretization(arrayin,stepnum)
% 
%   Input - 
%   arrayin: the input array;
%   stepnum: step number, which defines the discretization accuracy.
%   Output - 
%   arrayin: the output array after discreization.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 
% parameter check
narginchk(1,2);
stepnum = [];

if isvector(arrayin)==1
    arrayin = arrayin(:);
    narray = length(arrayin);
    arrayout = zeros(size(arrayin));
else
    error('The input parameter''arrayin''must be a 1-D array.');
end

switch numel(varargin)
    case 0
    case 1
        stepnum = varargin{:};
    otherwise
        error('Input parameter error!');
end

if isempty(stepnum)
    stepnum = 10;
end

%% discretization
pdiff = (max(arrayin)-min(arrayin))/stepnum;
for i=1:narray
    for level = 0:stepnum-1
        if arrayin(i)>=(min(arrayin)+level*pdiff) && arrayin(i)<(min(arrayin)+(level+1)*pdiff)
            arrayout(i)=level+1;
            break;
        end
    end
    if arrayout(i)==0               % this means arrayin(i)==max(arrayin)
        arrayout(i)=stepnum;
    end
end

end