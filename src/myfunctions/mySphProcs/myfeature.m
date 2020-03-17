function feature = myfeature(x,fs)
%MYFEATURE - Extract features of the signal x
%
%   feature = myfeature(x,fs)

%% 幅度归一化
x = x/max(abs(x));
%% 重采样
if fs ~=8000
    x = resample(x,8000,fs);
    fs = 8000;
end
%% 预加重
x = filter([1,-0.9375],1,x);

%% 预处理 - 提取浊音帧，除去其他帧
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
[frameA,st_energy,st_zerorate,~] = mytimefeature(x,fs,nwin,noverlap,[0,0]);
[~,~,~,x_voiced] = myendpointdetect(frameA,fs,st_energy,st_zerorate,[mean(st_energy),median(st_zerorate)],noverlap,0,0);
x_voiced = x_voiced/max(abs(x_voiced));

%% 语音波形图
% t1 = (0:length(x)-1)/fs;
% t2 = (0:length(x_voiced)-1)/fs;
% figure;
% subplot(1,2,1);plot(t1,x);xlabel('time/s');ylabel('amp');title('Original speech');
% subplot(1,2,2);plot(t2,x_voiced);xlabel('time/s');ylabel('amp');title('Voiced speech');

%% 提取特征
% 频域特征
freqfts = myfreqdomainfts(x_voiced,fs,nwin,noverlap);
freqfts = freqfts(:)';      % 转变为行向量
% MFCC特征
[mfcc,~,~] = mymfcc(x_voiced,fs);
% 由于每一段语音的mfcc特征是一个矩阵(帧数*mfcc阶数)，所以如果想要和 freqfts 
% 搭配使用，就必须将其转化为向量(1*mfcc阶数)
% 但是 dmfcc 不能简单地平均了事，因为dmfcc就是mfcc的差分，
% 简单平均 = (第一帧的mfcc-最后一帧的mfcc)/帧数
% 以下操作应当是适宜的:
meanmfcc = mean(mfcc);
nf = size(mfcc,1);
meandmfcc = mean( mfcc(1:round(nf/2),:) )-mean( mfcc(round(nf/2)+1:nf,:) );
dmmfcc = [meanmfcc meandmfcc];
%% 输出特征
feature = [freqfts dmmfcc];         % 输出为一个行向量
% feature = freqfts;
end