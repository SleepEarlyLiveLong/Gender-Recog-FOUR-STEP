function varargout = myvectorframing(varargin)
%MYFRAMING - Transfer a signal vector into a matrix of data frames
%
%   This MATLAB function transfers a signal into data frames 
%
%   frame = myframing(x,nwin)
%   frame = myframing(x,nwin,noverlap)
%   frame = myframing(...,option)

%   option can be 'truncation' or 'padding', the default value is
%   'truncation'.

%% 参数处理
% 检查参数数目
narginchk(2,4);
nargoutchk(0,1);

% 初始化输入参数
noverlap = [];
option = 'truncation';

% 获取输入参数option
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'}))
    option = varargin{end};
    varargin(end)=[];
end

% 获取剩余输入参数值
narg = numel(varargin); % 获取剩余输入参数数量
switch narg
    case 2
        [x,nwin] = varargin{:};
    case 3
        [x,nwin,noverlap] = varargin{:};
end

% 检查输入参数值
% 检查参数x
if isvector(x)==1
    x = x(:);   % 将x转为列向量
    nx = length(x);
else
    error('输入参数x必须为1维数组');
end
% 检查参数noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif (noverlap>=nwin) || (noverlap<0)
    error('noverlap数值必须小于nwin数值且为非负值');
end

%% 分帧预处理
% 帧移
nstride = nwin-noverlap; 
% 信号x不能分成整数帧，采用截断措施 
if strcmpi(option,'truncation')
    nframe = fix((nx-noverlap)/nstride);   %帧数
% 信号x不能分成整数帧，采用补零措施   
else
    nframe = ceil((nx-noverlap)/nstride);  %帧数
    npadding = nframe*nstride+noverlap-nx;
    x = [x;zeros(npadding,1)];  %末尾补零
end

%% 分帧
frame = zeros(nframe,nwin);
for i=1:nframe
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    frame(i,:) = x(start_index:end_index);
end

%% 返回分帧结果
varargout = {frame};

end