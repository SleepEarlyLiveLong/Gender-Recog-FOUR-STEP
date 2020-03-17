function fts4 = yourfeature4(voiced,fs)
%MYFEATURE1 - Extract YOURCUSTOM feature of the signal x
%
%   fts4 = myfeature4(x,fs)

%% 
% ·ÖÖ¡´úÂë
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
frame = myvectorframing(x_voiced,nwin,noverlap,'truncation');



fts4 = fts4(:)';
end