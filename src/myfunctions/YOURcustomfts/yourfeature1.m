function fts1 = yourfeature1(voiced,fs)
%MYFEATURE1 - Extract YOURCUSTOM feature of the signal x
%
%   fts1 = myfeature1(x,fs)

% 做一个写好的代码，加注释，解开即运行

%% 
% 分帧代码
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
frame = myvectorframing(x_voiced,nwin,noverlap,'truncation');



fts1 = fts1(:)';
end