function varargout = mydeframing(varargin)
%MYDEFRAMING - Transfer a matrix of data frames into a signal vector.
%
%   This MATLAB function transfers data frames into a signal 
%
%   x = mydeframing(frame)
%   x = mydeframing(frame,noverlap)

%% 参数处理
% 检查参数数目
narginchk(1,2);
nargoutchk(0,1);

% 初始化输入参数
noverlap = [];

% 获取输入参数值
switch nargin
    case 1
        frame = varargin{:};
    case 2
        [frame,noverlap] = varargin{:};
end

% 检查输入参数值
% 检查参数frame
if ismatrix(frame)
    [nframe,nwin] = size(frame);
else
    error('输入参数frame必须为矩阵');
end
% 检查参数noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif (noverlap>=nwin) || (noverlap<0)
    error('noverlap数值必须小于nwin数值且为非负值');
end

%% 合成x
% 帧移
nstride = nwin-noverlap; 
% x长度
nx = nframe*nstride+noverlap;
x1 = zeros(nx,1);
x2 = zeros(nx,1);
for i=1:nframe %保留后项
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    x1(start_index:end_index) = frame(i,:);
end
for j=1:nframe %保留前项
    i = nframe-j+1;
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    x2(start_index:end_index) = frame(i,:);
end
x = (x1+x2)/2;

%% 返回合成结果x
varargout = {x};
