function voiced = mygetvoiced(x,fs)
%GETVOICED - Get voiced sound of a certain voice segment 
%
%   voiced = getvoiced(x,fs)

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
[~,~,~,voiced] = myendpointdetect(frameA,fs,st_energy,st_zerorate,[mean(st_energy),median(st_zerorate)],noverlap,0,0);
voiced = voiced/max(abs(voiced));

end