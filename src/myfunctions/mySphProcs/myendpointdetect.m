function varargout = myendpointdetect(frame,fs,st_energy,st_zerorate,thresh,noverlap,pattern,disp)
%GETSPEECH - extract speech signal by the time domain feature
%
%   label = myendpointdetect(frame,fs,st_energy,st_zerorate,[thresm,thresz])
%   label = myendpointdetect(frame,fs,st_energy,st_zerorate,[thresm,thresz],noverlap)
%   label = myendpointdetect(frame,fs,st_energy,st_zerorate,[thresm,thresz],noverlap,pattern)
%   label = myendpointdetect(frame,fs,st_energy,st_zerorate,[thresm,thresz],noverlap,pattern,disp)
%   [label,endpoint,S1] = myendpointdetect(...)
%   [label,endpoint,S1.S2] = myendpointdetect(...)

%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% 参数数目检测
narginchk(5,8);
nargoutchk(0,4);

%% 参数初始化
%帧长、帧数
[nframe,nwin] = size(frame);
%帧重叠长度
if nargin<6
    noverlap = round(nwin/2);
end
%是否显示波形
if nargin<7
    disp = false;
end
%帧移
nstride = nwin-noverlap;

%% 筛选出超过阈值的帧
threshm = thresh(1);
threshz = thresh(2);
indexm = st_energy >= threshm;
indexz = st_zerorate >= threshz;
if pattern == 1   
    label = indexm|indexz;  % 检测语音(包括清音)
else
    label = indexm;         % 仅检测浊音
end

%% 寻找端点
current = 0;
endpoint_index = [];
for i=1:nframe
    last = current;
    current = label(i);
    if (last==0) && (current==1) %语音帧开始
        start_index = i;
    elseif (last==1) && (current==0) %语音帧结束
        end_index = i-1;
        index = false(nframe,1);
        index(start_index:end_index) = 1;
        if ~any(index&indexm) %误检的噪音帧
            label(start_index:end_index) = 0;
        else
            endpoint_index = cat(1,endpoint_index,[start_index,end_index]);
        end
    elseif (i==nframe) && (current==1) %语音帧结束
        end_index = i;
        index = false(nframe,1);
        index(start_index:end_index) = 1;
        if ~any(index&indexm) %误检的噪音帧
            label(start_index:end_index) = 0;
        else
            endpoint_index = cat(1,endpoint_index,[start_index,end_index]);  
        end
    end
end
% 非语音帧置0
frame(~label,:) = 0;
% 信号合成
S1 = mydeframing(frame,noverlap);
% 纯语音(不包含静音段)
signal_frame = frame;
signal_frame(~label,:) = [];
S2 = mydeframing(signal_frame,noverlap);

%% 端点位置
endpoint = zeros(size(endpoint_index));
endpoint(:,1) = (endpoint_index(:,1)-1)*nstride+1;
endpoint(:,2) = (endpoint_index(:,2)-1)*nstride+nwin;
% ---------- 在特定应用下可以鉴别虚标,即不是音节却被端点标出
% syllable_len = endpoint(:,2)-endpoint(:,1);
% syllable_num = length(syllable_len);
% if syllable_num > 1
%     % 认为小于最长音节1/3的语音段是虚标
%     Thresh = round(1/3*max(syllable_len));
%     for i=1:syllable_num
%         if syllable_len(i) < Thresh     
%             endpoint(i,:)=0;
%         end
%     end
% end
% % endpoint 的全0行删除
% endpoint(all(endpoint==0,2),:) = [];
% ---------- 士兰微合作项目,对语音内容有所考量 --------------------

%% 没有输出参数或disp==true则绘制信号波形
if nargout==0 || disp
    nx = length(S1);
    t = (0:(nx-1))/fs;
    plot(t,S1);
    hold on;
    xlabel('时间(s)');
    ylabel('幅度');
    title('提取的语音信号时域波形');
    for i=1:size(endpoint,1)
        x = (endpoint(i,1)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[1,0,0]);
        x = (endpoint(i,2)-1)/fs;
        line(x,0,'Marker','.','MarkerSize',20,'Color',[0,1,0]);
    end
    hold off;   
end

%% 输出
switch nargout
    case 1
        varargout = {label};
    case 2
        varargout = {label,endpoint};
    case 3
        varargout = {label,endpoint,S1};
    case 4
        varargout = {label,endpoint,S1,S2};
end

end