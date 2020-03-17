function freqfts = myfreqdomainfts(x_voiced,fs,nwin,noverlap)

% 分帧
frame = myvectorframing(x_voiced,nwin,noverlap,'truncation');
[nframe,framelen] = size(frame);
% 逐帧计算 频谱 和 倒谱
nfft=max(256,power(2,ceil(log2(framelen))));
amp = zeros(nframe,nfft);
c = zeros(nframe,nfft);
for i=1:nframe
    window = (hamming(framelen))';
    single_frame = frame(i,:).*window;
    amp(i,:) = abs(fft(single_frame,nfft));
    logamp = log(amp(i,:)+eps);     % 防止amp==0
    c(i,:) = real(ifft(logamp));
end

%% 
% figure;
% subplot(2,2,1);plot(frame(1,:));xlabel('sample');ylabel('amptitude');title('某一帧语音');
% subplot(2,2,2);plot(single_frame);xlabel('sample');ylabel('amptitude');title('该帧语音加窗后');
% subplot(2,2,3);plot(amp(1,:));xlabel('sample');ylabel('amptitude');title('该帧语音幅频谱');
% subplot(2,2,4);plot(c(1,:));xlabel('sample');ylabel('amptitude');title('该帧语音倒谱');

%% 特征
amp = amp(:,1:nfft/2);      % 截取对称的一部分
amp = amp./repmat(sum(amp,2),1,nfft/2);     % map矩阵行归一化
MF = zeros(nframe,0);
SE = zeros(nframe,0);
for i=1:nframe
    mf = 0;
    se = 0;
    for j=1:nfft/2
       mf = mf + j*amp(i,j);    
       se = se - amp(i,j)*(log(amp(i,j))/log(2));   
    end
    MF(i) = mf/nfft*fs;         % 某一帧的加权平均频率
    SE(i) = se/(nfft/2);        % 某一帧的谱熵
end
% 1 该段语音的频率均值(kHz)
fmean = mean(MF)/1000;
% meanfrenqucy = meanfreq(x_voiced,fs);
% 2 该段语音的频率标准差(kHz)
fstd = std(MF)/1000;
% 3 该段语音的频率中值(kHz)
fmid = median(MF)/1000;
% medianfrenqucy = medfreq(x_voiced,fs);
% 4 该段语音的频率Q25(kHz)
fQ25 = quantile(MF,0.25)/1000;
% 5 该段语音的频率Q75(kHz)
fQ75 = quantile(MF,0.75)/1000;
% 6 该段语音的频率四分位间距 (kHz)
fiqr = iqr(MF)/1000;
% 7 该段语音的斜偏
skew = skewness(MF);
% 8 该段语音的峰度
kurt = kurtosis(MF);
% 9 该段语音的平均谱熵
spent = mean(SE);
% 10 该段语音的谱熵光滑度 (多个小数连乘，数据容易变为0，舍弃此特征)
% sfm = prod(SE)^(1/nframe)/mean(SE);
% 11 该段语音的频率的众数
nstep = 16;
stats = mynumstatistic(mydiscretization(MF,nstep));
[~,pos] = max(stats(:,2));
mode = max(MF)*pos/nstep/1000;
% 基频(3 fts)(kHz)
[~,Pitch,~] = mypitchtrack(x_voiced,fs,nwin);
% 12 该段语音的平均基频
pitchmean = Pitch.mean/1000;
% 13 该段语音的最大基频
pitchmax = Pitch.max/1000;
% 14 该段语音的最小基频
pitchmin = Pitch.min/1000;

%% 输出
% 输出为一个行向量
freqfts = [fmean fstd fmid fQ25 fQ75 fiqr skew kurt spent mode ...
    pitchmean pitchmax pitchmin];

end