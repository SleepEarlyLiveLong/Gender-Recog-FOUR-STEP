function number = myhowmany(digit,matrix)
%MYHOWMANY - To count how many certain digit in the matrix.
%   
%   number = myhowmany(digit,matrix)
% 
%   Input - 
%   digit: the number to be counted;
%   matrix: the matrix to be checked.
%   Output - 
%   number: the number of the certain digit.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 
% parameter check
narginchk(1,2);

if ~ismatrix(matrix)
    error('Error! The second parameter should be a matrix.');
end
[m,n] = size(matrix);
if m==0 || n==0
    error('Error! You should as least put in something.');
end

%% do the statistics
count = 0;
for j=1:n
    for i=1:m
        if matrix(i,j)==digit
            count = count+1;
        end
    end
end
number = count;

end
%%