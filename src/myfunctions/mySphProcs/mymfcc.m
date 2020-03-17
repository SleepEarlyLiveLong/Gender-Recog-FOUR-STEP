function varargout = mymfcc(x,fs)
%MYMFCC - Mel-Frequency Cepstral Coefficients
%
%   mfc = mymfcc(x,fs)
%   [mfcc,dmfcc,ddmfcc] = mymfcc(x,fs)

%% 函数参数检查
narginchk(2,2);
nargoutchk(1,3);

%% 参数
n = 24; %Mel滤波器阶数
p = 12; %倒谱阶数
nwin = 256; %帧长

%% 获得Mel滤波器组
bank = melbankm(n,nwin,fs,0,0.5,'t'); 
% 归一化mel滤波器组系数
bank = full(bank);
bank = bank/max(bank(:));

%% 归一化倒谱提升窗口
w = 1+0.5*p*sin(pi*(1:p)./p);
w = w/max(w);

%% 预加重滤波器
x = double(x);
x = filter([1,-0.9375],1,x);

%% 语音信号分帧
frame = myvectorframing(x,nwin);
nframe = size(frame,1);
%% 计算每帧的MFCC参数
mfccmat = zeros(nframe,p+1);
for i=1:nframe
    y = frame(i,:)';
    y = y.* hamming(nwin); % 加窗
    energy = log(sum(y.^2)+eps); % 能量
    y = abs(fft(y));
    y = y(1:fix(nwin/2)+1);
    c = dct(log(bank*y+eps));   % Discrete cosine transform 离散余弦变换
    c = c(2:p+1)'.*w; % 取2~p+1个系数
    mfcc = [c,energy];
    mfccmat(i,:) = mfcc;
end

%% 一阶差分MFCC系数
dmfccmat = zeros(nframe,p+1);
for i=2:nframe-1
  dmfccmat(i,:) = mfccmat(i,:)-mfccmat(i-1,:);
end

%% 二阶差分MFCC系数
ddmfccmat = zeros(nframe,p+1);
for i=3:nframe-2
  ddmfccmat(i,:) = dmfccmat(i,:)-dmfccmat(i-1,:);
end

%% 合并MFCC及一阶、二阶差分系数
mfc = [mfccmat,dmfccmat,ddmfccmat];
%去除首尾各两帧，因为这两帧的二阶差分参数为0
mfc = mfc(3:nframe-2,:);

%% 输出参数
switch(nargout)
    case 1
        varargout = {mfc};
    case 2
        varargout = {mfccmat,dmfccmat};
    case 3
        varargout = {mfccmat,dmfccmat,ddmfccmat};
end
