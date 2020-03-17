%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              方案一                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function varargout = mypitchtrack(varargin)
% %PITCHTRACK - Pitch track by cepstrum analysis
% %
% %   This MATLAB function finds the pitch of the signal frames by 
% %   cepstrum analysis.
% %
% %   [period,T] = pitchtrack(frame,fs)
% %   [period,T] = pitchtrack(x,fs,nwin)
% 
% %% 参数处理
% % 检查参数数目
% narginchk(2,3);
% nargoutchk(1,2);
% 
% % 获取输入参数值
% % 获取第一个输入参数值
% arg1 = varargin{1};
% if isvector(arg1)   % arg1是向量x，输入还没有分帧
%     [x,fs,nwin] = varargin{:};
%     frame = framing(x,nwin,0,'truncation');
% else                % arg1是矩阵frame，输入已经分帧
%     [frame,fs] = varargin{:};
% end
% [nwin,nframe] = size(frame);
% 
% %% 计算倒谱
% nfft=max(256,power(2,ceil(log2(nwin))));  % fft变换点数
% cep=zeros(nfft,nframe);
% for i=1:nframe
%     cep(:,i)=myrceps(frame(:,i));
% end
% 
% %% 计算能量阈值(浊音判断)
% E=sum(frame.^2);        %按列求和
% % magnitude = sum(abs(frame));        % 每帧能量
% % threshmedian = median(magnitude);   % 中值
% % threshmean = mean(magnitude);       % 均值
% % if threshmean>1.5*threshmedian      % 如果均值和中值非常接近，则认为大部分均为语音信号，阈值设为0
% %     threshe = threshmedian;
% % else
% %     threshe = 0;
% % end
% 
% %% 提取浊音，计算基音周期 ->frame_period
% % 倒谱的横坐标量纲是时间t，人的基音：2-20ms/50-500Hz
% ncep_start = round(0.002*fs+1);                  % 倒谱开始点
% ncep_end = min(round(0.02*fs+1),round(nwin/2));  % 倒谱结束点
% cep_max=zeros(1,nframe);                         % 每一帧倒谱最大值
% frame_period = zeros(1,nframe);                  % 每一帧的基音周期(每一帧倒谱最大值点)
% % 计算一段语音的能量均值和方差，判断是不是纯浊音
% % 整段非浊音(噪音/清音)
% if mean(E)<0.5 && var(E)<0.01        % 0.5和0.01都是经验数值   
%     for i=1:nframe
%         frame_period(i)=0;           % 不考虑基音频率，令每一帧基频为0
%     end
% % 整段纯浊音，阈值为0
% elseif mean(E)>=2 && var(E)<1        % 2和1也都是经验数值                           
%     for i=1:nframe                   % 不存在一帧能量不够和倒谱无明显峰值
%         [cep_max(i),frame_period(i)]=max(cep(ncep_start:ncep_end,i));
%     end
% % 包含了V/U/S
% else
%     threshe=median(E);      % 根据语音的特性取中值为阈值
%     for i=1:nframe
%         if E(i)<threshe              % 能量不够，不是浊音(很可能是静音)
%             frame_period(i)=0;
%         else
%             [cep_max(i),frame_period(i)]=max(cep(ncep_start:ncep_end,i));
%             threshild=4*abs(mean(cep(ncep_start:ncep_end,i)));
%             if cep_max(i)<threshild   % 倒谱没有明显的峰值，不是浊音(很可能是清音)
%                 frame_period(i)=0;
%             end
%         end
%     end
% end
% % 转以时间为单位
% frame_period=(frame_period+ncep_start-2)/fs;    %注意！这里是-2不是-1
% for i=1:nframe
%     if frame_period(1,i)<0.002        %(1/8000=0.000125),所以留下的都是0.001875
%         frame_period(1,i)=0;
%     end
% end
% 
% %% 计算每一帧的中间点时间(T)
% T = zeros(1,nframe);
% for i=1:nframe
%     start_time = (i-1)*nwin;
%     T(i) = start_time+nwin/2;
% end
% T = T/fs;
% 
% %% 返回输出结果
% if nargout==1
%     varargout = {frame_period};
% else
%     varargout = {frame_period,T};
% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              方案二                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function varargout = mypitchtrack(varargin)
%PITCHTRACK - Pitch track by cepstrum analysis
%
%   This MATLAB function finds the pitch of the signal frames by 
%   cepstrum analysis.
%
%   [period,pitch,T] = pitchtrack(frame,fs)
%   [period,pitch,T] = pitchtrack(x,fs,nwin)

%% 参数处理
% 检查参数数目
narginchk(2,3);
nargoutchk(1,3);

% 获取输入参数值
% 获取第一个输入参数值
arg1 = varargin{1};
if isvector(arg1)   % arg1是向量x
    [x,fs,nwin] = varargin{:};
    frame = myvectorframing(x,nwin,0,'truncation');
else                % arg1是矩阵frame
    [frame,fs] = varargin{:};
end
[nframe,~] = size(frame);

%% 计算能量阈值
magnitude = sum(abs(frame),2);      % 每帧能量
threshmedian = median(magnitude);   % 中值
threshmean = mean(magnitude);       % 均值
if threshmean>1.5*threshmedian      % 如果均值和中值非常接近，则认为大部分均为语音信号，阈值设为0
    threshe = threshmedian;
else
    threshe = 0;
end

%% 计算基音周期(向量)
%利用人的基音周期范围2~20ms(50~500Hz)
tstart = round(0.002*fs+1);
tend = min(round(0.02*fs+1),round(nwin/2));
period = zeros(1,nframe);
for i=1:nframe
    if magnitude(i)>=threshe
        c = myrceps(frame(i,:));        % 计算某一帧的倒谱
        [maximum,maxpos] = max(c(tstart:tend));
        threshold = 4*mean(abs(c(tstart:tend)));
        if maximum>=threshold       %浊音
            period(i) = (maxpos+tstart-2)/fs;
        else                        %清音或静音
            period(i) = 0;
        end
    else                            %清音或静音
        period(i) = 0;
    end
end

%% 计算一段语音的平均基音频率 pitch 
pitchs = 1./period;           
pitchs(pitchs==Inf) = [];
Pitch.mean = mean(pitchs);
Pitch.min = min(pitchs);
Pitch.max = max(pitchs);

%% 计算每一帧的中间点时间(T)
T = zeros(1,nframe);
for i=1:nframe
    start_time = (i-1)*nwin;
    T(i) = start_time+nwin/2;
end
T = T/fs;

%% 输出
switch nargout
    case 1
        varargout = {period};
    case 2
        varargout = {period,Pitch};
    case 3
        varargout = {period,Pitch,T};
end

end