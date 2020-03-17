function [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs,varargin)
%MYTIMEFEATURE - short-time time domain features of a sector of voice
%
%   This MATLAB function returns [frame,st_energy,st_zerorate], the short-time 
%   energy and zero rate of the input signal vector x.
%
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs)
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs,nwin)
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs,nwin,noverlap)
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs,nwin,noverlap,[threshm,threshz])
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(x,fs,nwin,noverlap,[threshm,threshz],disp)
%   [frame,st_energy,st_zerorate,T_series] = mytimefeature(...,option)
%   mytimefeature(...)

%   option：{'truncation'},'padding'  

%% 参数数目检测
narginchk(2,7);
nargoutchk(0,4);

%% 参数获取
%截断、补零选择
option = 'truncation';
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'}))
    option = varargin{end};
    varargin(end)=[];
end
%获取剩余输入参数数目
narg = numel(varargin);
%定义参数
ntime = 20;     % 20ms
nwin = fs*ntime/1000;
noverlap = round(nwin/2);
disp = false;
thresh = [0,0];
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
        error('输入参数不对\n');
end

%% 参数值检测
%将x转为列向量
if isvector(x)==1
    x = x(:);
else
    error('输入参数x必须为1维数组\n');
end
%帧重叠长度noverlap
if noverlap >= nwin
    error('参数noverlap数值必须小于参数window的数值\n');
end
%阈值thresh
threshm = 0;       % 短时平均幅度阈值
threshz = 0;       % 短时过零率阈值
if length(thresh)==2
    threshm = thresh(1);
    threshz = thresh(2);
end

%% 分帧
frame = myvectorframing(x,nwin,noverlap,option);
nframe = size(frame,1);
for i=1:nframe
    %去直流分量
    frame(i,:) = frame(i,:)-mean(frame(i,:));
end

%% 计算时域特征
%短时平均幅度(表示能量)
st_energy = zeros(nframe,1);
for i=1:nframe
    st_energy(i) = sum(abs(frame(i,:)));
end
%短时过零率
st_zerorate = zeros(nframe,1);
for i=1:nframe
    st_zerorate(i) = 0.5*sum(abs(sign(frame(i,2:nwin))-sign(frame(i,1:(nwin-1)))));
end

%% 计算每一帧的中间点时间(T)
nstride = nwin-noverlap;
T_series=zeros(nframe,1);
for i=1:nframe
    start_time=(i-1)*nstride;
    T_series(i)=start_time+nwin/2;
end
T_series=T_series/fs;

%% 没有输出参数或disp==true则绘制信号波形
if nargout==0 || disp
    figure;
    plot(T_series,st_energy,'r',T_series,st_zerorate,'g');
    strlegend = {'短时平均幅度','短时过零率'};
    hold on;
    if threshm~=0
        ThM = T_series;
        ThM(:) = threshm;
        plot(T_series,ThM,'r','LineWidth',2);
        strlegend = cat(2,strlegend,'短时平均幅度阈值');
    end  
    if threshz~=0
        ThZ = T_series;
        ThZ(:) = threshz;
        plot(T_series,ThZ,'g','LineWidth',2);
        strlegend = cat(2,strlegend,'短时过零率阈值');
    end
    hold off;
    legend(strlegend);
    xlabel('时间(s)');
    ylabel('幅度');
    title('时域波形');
end

end