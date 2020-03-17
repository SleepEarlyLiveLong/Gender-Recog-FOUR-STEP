function varargout = myrowcheck(mA,mB)
%MYROWCHECK - The data discreization function.
%   
%   arrayout = myrowcheck(mA,mB)
%   [array_row,samerow_A] = myrowcheck(mA,mB)
%   [array_row,samerow_A,samerow_B] = myrowcheck(mA,mB)
% 
%   Input - 
%   mA: the input matrix A;
%   mB: the input matrix B.
%   Output - 
%   array_row: the output array with the serial number of same rows of
%               these 2 matrices.
%   samerow_A: number of same rows of matrix A(must have 2 columns);
%   samerow_B: number of same rows of matrix B(must have 2 columns);
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 
% parameter check
narginchk(2,2);
nargoutchk(1,3);

if ~ismatrix(mA) || ~ismatrix(mB) || size(mA,2)~=size(mB,2)
    error('Input error!');
end

% the algorithm
row_A = size(mA,1);
row_B = size(mB,1);
array_row=zeros(max(row_A,row_B),2);
samerow_A = zeros(0,0);
samerow_B = zeros(0,0);

lineA=0;
for i=1:row_A
    for j=i+1:row_A
        if mA(i,:)==mA(j,:)
            lineA = lineA+1;
            samerow_A(lineA,:) = [i,j];
        end
    end
end
lineB=0;
for i=1:row_B
    for j=i+1:row_B
        if mB(i,:)==mB(j,:)
            lineB = lineB+1;
            samerow_B(lineB,:) = [i,j];
        end
    end
end

k=0;
for i=1:row_A
    for j=1:row_B
        if mA(i,:)==mB(j,:)
            k=k+1;
            array_row(k,1)=i;
            array_row(k,2)=j;
        end
    end
end

if k < size(array_row,1)
    for i=(k+1):size(array_row,1)
        array_row(k+1,:)=[];
    end
end
            
switch nargout
    case 1
        varargout = {array_row};
    case 2
        varargout = {array_row,samerow_A};
    case 3
        varargout = {array_row,samerow_A,samerow_B};
end

end
%% 