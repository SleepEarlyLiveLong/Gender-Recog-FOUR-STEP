function c = myrceps(single_frame)
% MYRCEPS - Calculate the cepstrum
%
%   c = myrceps(x,nfft)

%% 参数预判
if ~isvector(single_frame)
    error('Error! Input parameter "single_frame" should be a vector\n');
end
single_frame = single_frame(:);
%% 计算
% 加窗
window = hamming(length(single_frame));
single_frame = single_frame.*window;
% FFT 得到频谱
nfft=max(256,power(2,ceil(log2(length(single_frame)))));
amp = abs(fft(single_frame,nfft));
% 得到对数频谱
logamp = log(amp+eps);     % 防止amp==0
% 计算倒谱
c = real(ifft(logamp));

end