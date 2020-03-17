function varargout = mygetspeech(frame,fs,M,Z,threshm,threshz,noverlap,disp)
%GETSPEECH - extract speech signal by the time domain feature
%
%   S = getspeech(frame,fs,M,Z,thresm,thresz)
%   S = getspeech(frame,fs,M,Z,thresm,thresz,noverlap)
%   S = getspeech(frame,fs,M,Z,thresm,thresz,noverlap,disp)
%   [S,startp,endp] = getspeech(...)

%参数数目检测
narginchk(6,8);
nargoutchk(0,3);

%%
%参数初始化
%帧长、帧数
[nwin,nframe] = size(frame);
%帧重叠长度
if nargin<7
    noverlap = round(nwin/2);
end
%是否显示波形
if nargin<8
    disp = true;
end
%帧移
nstride=nwin-noverlap;
%信号长度
nx = nframe*nstride+noverlap;
S = zeros(nx,1);

%%
%筛选出小于阈值的帧
indexm = M < threshm;
indexz = Z < threshz;
%小于阈值的帧设为0
frame(:,indexm&indexz) = 0;

%%
%信号合成
startp = [];
endp = [];
isstart = false;
for i=1:nframe
    start_index = (i-1)*nstride+1;
    mid_index =  start_index+nstride-1;
    end_index = start_index+nwin-1;
    %非零帧的值才赋给S
    if frame(1,i)~=0
        S(start_index:end_index) = frame(:,i);
        %语音段起始点
        if ~isstart
            startp = [startp,start_index];
            isstart = true;
        end
        %末尾为语音，则加一个终止点
        if i==nframe
            endp = [endp,end_index];
        end
    else
        %语音段终止点
        if isstart
            endp = [endp,mid_index];
            isstart = false;
        end
    end
end

%%
%没有输出参数或disp==true则绘制信号波形
if nargout==0 || disp
     t = (0:(nx-1))/fs;
    plot(t,S);
    hold on;
    xlabel('时间(s)');
    ylabel('幅度');
    title('提取的语音信号时域波形');
    for i=1:length(startp)
        x = (startp(i)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[1,0,0]);
        x = (endp(i)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[0,1,0]);
    end
    hold off;   
end

switch nargout
    case 1
        varargout = {S};
    case 2
        varargout = {S,startp};
    case 3
        varargout = {S,startp,endp};
end


