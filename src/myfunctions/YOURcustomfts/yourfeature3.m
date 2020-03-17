function fts3 = yourfeature3(voiced,fs)
%MYFEATURE3 - Extract YOURCUSTOM feature of the signal x
%
%   fts3 = myfeature3(x,fs)

%% 
% ·ÖÖ¡´úÂë
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
frame = myvectorframing(x_voiced,nwin,noverlap,'truncation');




fts3 = fts3(:)';
end