function [int_num,nointxy] = myisinterger(matrix)
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
[m,n]=size(matrix);
count = 0;
k=1;
array = zeros(1,2);
for j=1:n
    for i=1:m
        if fix(matrix(i,j)) == matrix(i,j)
            count = count+1;
        else
            array(k,1)=i;
            array(k,2)=j;
            k=k+1;
        end
    end
end

int_num = count;
nointxy = array;
end
%%