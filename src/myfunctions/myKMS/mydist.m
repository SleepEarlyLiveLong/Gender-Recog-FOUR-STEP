function dists = mydist(distance,temp,C)
% 此函数纯粹是为了简化主代码-mykmeans.m

%% 
if ~ischar(distance)
    error('Error! "distance" should be string.');
end
if ~ismatrix(temp) || ~ismatrix(C)
    error('Error! You should put in 2 matrics.');
end
if size(temp)~=size(C)
    error('Error! You should put in 2 matrics..');
end

if strcmp(distance,'sqEuclidean')
    dists = sqrt(sum( (temp-C).^2 ,2));
elseif strcmp(distance,'cityblock')
    dists = sum( abs((temp-C)) ,2);
elseif strcmp(distance,'cosine')
    dists = mydist_cosine(temp,C);
elseif strcmp(distance,'correlation')
    dists = mydist_corre(temp,C);
elseif strcmp(distance,'Hamming')
    dists = mydist_hamm(temp,C);
end

end
%%