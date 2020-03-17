function err = myerrcal(distance,X,C)
% 此函数纯粹是为了简化主代码-mykmeans.m

%%
if ~ischar(distance)
    error('Error! "distance" should be string.');
end
if ~ismatrix(X) || ~ismatrix(C)
    error('Error! You should put in 2 matrics.');
end

[N,~] = size(X);
[~,P] = size(C);
if size(X,2)~=size(C,2)+1
    error('Error! You should put in 2 matrics with correct size.');
end

err = 0;
if strcmp(distance,'sqEuclidean')
    for i=1:N
        err = err+sqrt( sum( (X(i,1:P)-C(X(i,P+1),:)).^2 ) );
    end
elseif strcmp(distance,'cityblock')
    for i=1:N
        err = err+sum( abs( X(i,1:P)-C(X(i,P+1),:) ) );
    end
elseif strcmp(distance,'cosine')
    for i=1:N
        err = err+mydist_cosine( X(i,1:P),C(X(i,P+1),:) );
    end
elseif strcmp(distance,'correlation')
    for i=1:N
        err = err+mydist_corre( X(i,1:P),C(X(i,P+1),:) );
    end
elseif strcmp(distance,'Hamming')
    for i=1:N
        err = err+mydist_hamm( X(i,1:P),C(X(i,P+1),:) );
    end
end

end
%% 