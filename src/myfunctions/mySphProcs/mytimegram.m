function [frame,E,M,Z,T] = mytimegram(x,fs,varargin)
%MYTIMEGRAM - short-time time domain freture
%
%   This MATLAB function returns [E,M,Z], the short-time time domain freture of the input
%   signal vector x.
%
%   [frame,E,M,Z,T] = mytimegram(x,fs)
%   [frame,E,M,Z,T] = mytimegram(x,fs,nwin)
%   [frame,E,M,Z,T] = mytimegram(x,fs,nwin,noverlap)
%   [frame,E,M,Z,T] = mytimegram(x,fs,nwin,noverlap,[threshe,threshm,threshz])
%   [frame,E,M,Z,T] = mytimegram(x,fs,nwin,noverlap,[threshe,threshm,threshz],disp)
%   [frame,E,M,Z,T] = myspectrogram(...,Property)
%   myspectrogram(...)

%   Property：{'truncation'},'padding'  

%参数数目检测
narginchk(2,7);

%将x转为列向量
if isvector(x)==1
    x = x(:);
    nx = length(x);
else
    error('输入参数''x''必须为1维数组');
end

%截断、补零选择
endprocess = 'truncation';
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'}))
    endprocess = varargin{end};
    varargin(end)=[];
end

%获取剩余输入参数数目
narg = numel(varargin);
%定义参数
nwin = 160;     %20ms (fs=8000Hz)
noverlap = round(nwin/2);
disp = true;
thresh = [0,0,0];
%获取输入参数值
switch narg
    case 0
    case 1
        nwin = varargin{:};      
    case 2
        [nwin,noverlap] = varargin{:};
    case 3
        [nwin,noverlap,thresh] = varargin{:};
    case 4
        [nwin,noverlap,thresh,disp] = varargin{:};
    otherwise
        error('输入参数不对');
end

%%
%参数检测
%帧重叠长度noverlap
if noverlap >= nwin
    error('''noverlap''数值必须小于''window''的长度');
end

%阈值thresh
threshe = 0;
threshm = 0;
threshz = 0;
if length(thresh)==3
    threshe = thresh(1);
    threshm = thresh(2);
    threshz = thresh(3);
end

%帧移nstride
nstride=nwin-noverlap; 
%信号x不能分成整数帧，采用截断措施 
if strcmpi(endprocess,'truncation')
    %帧数
    nframe=fix((nx-noverlap)/nstride);   
%信号x不能分成整数帧，采用补零措施   
else
    %帧数
    nframe=ceil((nx-noverlap)/nstride); 
    npadding=nframe*nstride+noverlap-nx;
    %末尾补零
    x=[x;zeros(npadding,1)];  
end

%%
%分帧
frame=zeros(nwin,nframe);
for i=1:nframe
    start_index=(i-1)*nstride+1;
    end_index=start_index+nwin-1;
    frame(:,i)=x(start_index:end_index);
    %去直流分类
    frame(:,i) = frame(:,i)-median(frame(:,i));
end

%%
%计算时域特征
%短时能量
E = zeros(1,nframe);
for i=1:nframe
    E(i) = sum((frame(:,i)).^2);
end

%短时平均幅度
M = zeros(1,nframe);
for i=1:nframe
    M(i) = sum(abs(frame(:,i)));
end

%短时过零率
Z = zeros(1,nframe);
for i=1:nframe
    Z(i) = 0.5*sum(abs(sign(frame(2:nwin,i))-sign(frame(1:(nwin-1),i))));
end

%计算每一帧的中间点时间(T)
T=zeros(1,nframe);
for i=1:nframe
    start_time=(i-1)*nstride;
    T(i)=start_time+nwin/2;
end
T=T/fs;

%没有输出参数或disp==true则绘制信号波形
if nargout==0 || disp
    plot(T,E,'r',T,M,'g',T,Z,'b');
    strlegend = {'短时能量','短时平均幅度','短时过零率'};
    hold on;
    if threshe~=0
        ThE = T;
        ThE(:) = threshe;
        plot(T,ThE,'r','LineWidth',2);
        strlegend = cat(2,strlegend,'短时能量阈值');
    end
    if threshm~=0
        ThM = T;
        ThM(:) = threshm;
        plot(T,ThM,'g','LineWidth',2);
        strlegend = cat(2,strlegend,'短时平均幅度阈值');
    end  
    if threshz~=0
        ThZ = T;
        ThZ(:) = threshz;
        plot(T,ThZ,'b','LineWidth',2);
        strlegend = cat(2,strlegend,'短时过零率阈值');
    end
    hold off;
    legend(strlegend);
    xlabel('时间(s)');
    ylabel('幅度');
    title('时域波形');
end




