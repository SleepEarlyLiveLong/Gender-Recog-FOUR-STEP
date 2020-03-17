function centers = mycluster_plus(X,k)
%MYCLUSTER_PLUS - K-means clustering initialization: kmeans++.
%   To initialize the cluster centers using kmeans++.
%   
%   centers = mycluster_plus(X,k)
% 
%   Input - 
%   X: the input N*P matrix X with N points of P-dimension;
%   k: the number of classes;
%   Output - 
%   centers: the initialized cluster centers.

%% 
if ~ismatrix(X) || ~isreal(k)
    error('Input parameter error! ');
end
[m,~] = size(X);
if k>m
    error('Error! Too many clustering classes.');
end

% specify the first center
B = [];
% choose the first point randomly
idx = mod(round(rand()*m),m)+1; 
B = cat(1,B,X(idx,:));          %存入B中
X(idx,:) = [];                  %从X中删除
%% 
while ~isempty(X)               %循环排序，直到X为空
    % 计算X中剩余点到B中所有点的距离和
    m = size(X,1);
    bn = size(B,1);
    dists = zeros(m,1);
    for i=1:m
        Point = X(i,:); %取出第i个点
        Mat = repmat(Point,bn,1); %扩展为bn行，方便矩阵相减
        diff = Mat-B;
        dist = sqrt(sum(diff.^2,2)); %沿第2维求平方和，再开根号，dist各行为Point到B各行的距离
        dists(i) = sum(dist); %求Point到B中所有点的距离之和
    end
    [~,idx] = max(dists);       %找最大值
    B = cat(1,B,X(idx,:));      %存入B中
    X(idx,:) = [];              %从X中删除
end
centers = B(1:k,:);

end
%%