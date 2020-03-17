function [kbest,Kbest] = myKKNNcgForClass

% 时间太紧张，等有空了再写
% 明确本函数的目的:
%   针对 kmeans+KNN 算法进行网络参数寻优，待确定的参数有聚类中心
%   的个数k 和 KNN 的K
% 明确技术路线:
%   输入参数应当只有训练集，这不关测试集的事。在训练集内部做 cross-validation
%   （一般取3-5），以此来确定 kbest 和 Kbest 两个参数的取值。