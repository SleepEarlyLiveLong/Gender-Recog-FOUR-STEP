function fts2 = yourfeature2(voiced,fs)
%MYFEATURE2 - Extract YOURCUSTOM feature of the signal x
%
%   fts2 = myfeature2(x,fs)

%% 
% ·ÖÖ¡´úÂë
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
frame = myvectorframing(x_voiced,nwin,noverlap,'truncation');



fts2 = fts2(:)';
end