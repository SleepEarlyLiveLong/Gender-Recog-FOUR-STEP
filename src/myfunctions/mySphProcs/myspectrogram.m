function varargout = myspectrogram(x,varargin)
%MYSPECTROGRAM - Spectrogram using a Short-Time Fourier Transform (STFT)
%
%   This MATLAB function returns S, the short time Fourier transform of the input
%   signal vector x.
%
%   S = myspectrogram(x)
%   S = myspectrogram(x,window)
%   S = myspectrogram(x,window,noverlap)
%   S = myspectrogram(x,window,noverlap,nfft)
%   S = myspectrogram(x,window,noverlap,nfft,fs)
%   S = myspectrogram(x,window,noverlap,nfft,fs,[fmin,fmax])
%   S = myspectrogram(x,window,noverlap,nfft,fs,[fmin,fmax,pmin,pmax])
%   [S,F,T,P] = myspectrogram(...)
%   myspectrogram(...)
%   myspectrogram(...,Property)

%   Property：{'xaxis'},'yaxis',{'color'},'gray',{'truncation'},'padding',{'spectall'},'spectpeaks'          

%参数数目检测
narginchk(1,10);
nargoutchk(0,4);

%将x转为列向量
if isvector(x)==1
    x = x(:);
    nx = length(x);
else
    error('输入参数''x''必须为1维数组');
end

%语谱显示全部、只选择峰值选择
spectmode = 'spectall';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'spectall','spectpeaks'}))
    spectmode = varargin{end};
    varargin(end)=[];
end

%截断、补零选择
endprocess = 'truncation';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'}))
    endprocess = varargin{end};
    varargin(end)=[];
end

%语谱图颜色模式
colormode = 'jet';
colorset = {'jet','hsv','hot','gray',...
            'cool','spring','summer','autumn',...
            'winter','gray','bone','copper',...
            'pink','lines'};
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},colorset))
    colormode = varargin{end};
    varargin(end)=[];
end

%频率轴设置，默认在x轴
freqloc = 'xaxis';
if (nargin > 1 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'yaxis','xaxis'}))
    freqloc = varargin{end};
    varargin(end)=[];
end

%获取剩余输入参数数目
narg = numel(varargin);
%初始化参数
window = [];
noverlap = [];
nfft = [];
fs = [];
lim = [];
%获取输入参数值
switch narg
    case 0
    case 1
        window = varargin{:};      
    case 2
        [window,noverlap] = varargin{:};
    case 3
        [window,noverlap,nfft] = varargin{:};
    case 4
        [window,noverlap,nfft,fs] = varargin{:};
    case 5
        [window,noverlap,nfft,fs,lim] = varargin{:};
    otherwise
        error('输入参数不对');
end

%初始化窗函数win
%没有给定窗长时，默认将信号分为8帧
if isempty(window)
    if strcmpi(endprocess,'truncation')
        win = hamming(fix(nx*2/9));
    else
        win = hamming(ceil(nx*2/9));
    end
%输入一个标量    
elseif isscalar(window)==1
    win = hamming(window);
%输入一个窗矢量    
else
    win = window(:);
end
%帧长
nwin = length(win);

%初始化noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif noverlap >= nwin
    error('''noverlap''数值必须小于''window''的长度');
end
 
%初始化nfft
if isempty(nfft)        
   nfft = max(256,power(2,ceil(log2(nwin))));
elseif nfft < nwin
   nfft =  power(2,ceil(log2(nwin)));
end

%初始化fs
%是否使用归一化频率
isFsnormalized = false;
if isempty(fs)    
    fs = 2*pi;
    isFsnormalized = true;
end

%初始化limit
if length(lim)==4
    flim = [lim(1),lim(2)];
    plim = [lim(3),lim(4)];
elseif length(lim)==2
    flim = [lim(1),lim(2)];
    plim = [];
else
    flim = [];
    plim = [];
end

%帧移
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

%分帧
frame=zeros(nwin,nframe);
for i=1:nframe
    start_index=(i-1)*nstride+1;
    end_index=start_index+nwin-1;
    %加窗
    frame(:,i)=x(start_index:end_index).*win;
end

%短时傅里叶变换的频点(F)
F = 0:fs/nfft:fs/2;
F = F';

%计算短时傅里叶变换(S)
nfreq = length(F);
S=zeros(nfreq,nframe);
for i=1:nframe
    X=fft(frame(:,i),nfft);
    S(:,i)=X(1:nfreq);
end

%计算每一帧的中间点时间(T)
T=zeros(1,nframe);
for i=1:nframe
    start_time=(i-1)*nstride;
    T(i)=start_time+nwin/2;
end
T=T/fs;
    
%计算能量谱密度(P)
P = zeros(nfreq,nframe);
%窗能量
Ewin = sum(win.^2);
%抵消窗能量的权重系数
k = zeros(nfreq,1);
k(:) = 2/(fs*Ewin);
%在0频和fs/2处，分子为1，其他频率分子为2
k(1) = 1/(fs*Ewin);
if F(end)==fs/2
    k(end) = 1/(fs*Ewin);
end
for i=1:nframe
    P(:,i) = k.*(abs(S(:,i)).^2);
end

%语谱只显示峰值
if strcmpi(spectmode,'spectpeaks')
    %计算每帧的能量峰值
    Pframemax = max(P);
    for i=1:nframe
        %非峰值引索
        index = P(:,i)~=Pframemax(i); 
        %非峰值处能量设置0
        P(index,i) = 0;
    end
end
    

%没有输出参数则绘制语谱图
if nargout==0
    %使用归一化频率
    if isFsnormalized
        F = F/pi;
        T = T*2*pi;
        flbl = '归一化频率 (\times\pi rad/sample)';
        tlbl = '时间 (sample)';
    else
        flbl = '频率 (Hz)';
        tlbl = '时间 (s)';
    end
    LogP = 10*log10(P+eps);
    if strcmpi(freqloc,'yaxis')
        surf(T,F,LogP,'edgecolor','none');
        xlbl = tlbl;
        ylbl = flbl;
    else
        surf(F,T,LogP','edgecolor','none');
        xlbl = flbl;
        ylbl = tlbl;
    end
    axis xy; 
    axis tight;
    view(0,90);
    colormap(colormode);
    xlabel(xlbl);
    ylabel(ylbl);
    title('语谱图');
    %设置能量显示范围
    if ~isempty(plim)
        zlim(plim); 
    end
    
    %设置频率显示范围
    if ~isempty(flim)
        if strcmpi(freqloc,'yaxis') 
            ylim(flim); 
        else
            xlim(flim); 
        end        
    end 
else
    switch nargout
        case 1
            varargout = {S};
        case 2
            varargout = {S,F};
        case 3
            varargout = {S,F,T};
        case 4
            varargout = {S,F,T,P};
    end
end
            
end    