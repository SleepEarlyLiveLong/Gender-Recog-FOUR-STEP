% Name:     GenderRecogGUI 0.3
% Function: Recognition of speaker's gender based on his/her speech signal.
%           The COMPLETE version of GenderRecogGUI 0.1
%           Add some more features

% Copyright (c) 2019 CHEN Tianyang
% more info contact: tychen@whu.edu.cn

%%
function varargout = GenderRecogGUI(varargin)
% GENDERRECOGGUI MATLAB code for GenderRecogGUI.fig
%      GENDERRECOGGUI, by itself, creates a new GENDERRECOGGUI or raises the existing
%      singleton*.
%
%      H = GENDERRECOGGUI returns the handle to a new GENDERRECOGGUI or the handle to
%      the existing singleton*.
%
%      GENDERRECOGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GENDERRECOGGUI.M with the given input arguments.
%
%      GENDERRECOGGUI('Property','Value',...) creates a new GENDERRECOGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GenderRecogGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GenderRecogGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menusettings.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GenderRecogGUI

% Last Modified by GUIDE v2.5 20-Mar-2019 17:09:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GenderRecogGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GenderRecogGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GenderRecogGUI is made visible.
function GenderRecogGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GenderRecogGUI (see VARARGIN)

% Choose default command line output for GenderRecogGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GenderRecogGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);
% 一些初始化内容放在这里
% 添加函数路径
addpath(genpath([pwd,'\myfunctions']));
addpath(genpath([pwd,'\libsvm322']));
global control;
control.classifier = 'DistCompare';        % 分类器名称
control.fold_list = [];         % 数据集所在文件夹
control.ftsstruct = [];         % 数据集的特征所构成的结构体
control.traindata = [];         % 训练集数据
control.trainlabel = [];        % 训练集标签
control.valdata = [];           % 验证集数据
control.vallabel = [];          % 验证集标签
control.ftsRange = 20;          % 默认NB的特征量化范围是0-20整数
control.clusters = 7;           % 默认KMSKNN的聚类中心数量是7
control.Knearest = 1;           % 默认KNN的最邻近范围是1
control.ValCorrRate = [];       % 验证集-准确率
control.RecallRate = [];        % 验证集-召回率
control.TrainSegRatio = 0.7;    % 默认训练集/验证集=7/3
control.datasetpath = [];       % 数据集地址[ones(1,13) zeros(1,26)]
control.ftsCandidateChain = [ones(1,13) zeros(1,26)];   % 特征候选表(1*39)
control.featurechanged = zeros(1,15);       % 特征是否已经改变 0-没有改变 1-已经改变
control.CurrentSplitRatioChoice = 3;               % 默认选7/3的分割比例
control.CurrentClassifierChoice = 1;               % 默认分类器 DistCompare
control.PastSplitRatioChoice = 3;
control.PastClassifierChoice = 1;
global param;
param.fs = 8000;
param.label = 'RecordedSound';
param.startindex = 1;
param.number = 0;
global recorder;
recorder.begin = false;           % 默认没有开始录音
recorder.voice = [];
recorder.R = audiorecorder(param.fs,16,1);
global classifierparam;
classifierparam.DCtraincore = [];       % DistCompare 的训练集中心(向量)
classifierparam.NBquantizer = [];
classifierparam.NBTrainingSets = [];
classifierparam.NBValidationSets = [];

% 初始化一些显示参数
set(handles.textCurrentAccuracy,'string',num2str(0,'%.3f'));
set(handles.textCurrentRecallRate,'string',num2str(0,'%.3f'));
% 初始化默认只选择13个频域特征
set(handles.checkbox_meanfreq,'Value',1);
set(handles.checkbox_freqstd,'Value',1);
set(handles.checkbox_midfreq,'Value',1);
set(handles.checkbox_Q25,'Value',1);
set(handles.checkbox_Q75,'Value',1);
set(handles.checkbox_IRQ,'Value',1);
set(handles.checkbox_skew,'Value',1);
set(handles.checkbox_kurt,'Value',1);
set(handles.checkbox_spent,'Value',1);
set(handles.checkbox_mode,'Value',1);
set(handles.checkbox_meanpitch,'Value',1);
set(handles.checkbox_maxpitch,'Value',1);
set(handles.checkbox_minpitch,'Value',1);
set(handles.checkbox_mfcc,'Value',0);
set(handles.checkbox_dmfcc,'Value',0);
% 默认比例是7:3
set(handles.MenuSeventyThirty,'Checked','on');
% 默认分类器是 DistCompare 简单举例判别
set(handles.MenuDistCompare,'Checked','on');

% --- Outputs from this function are returned to the command line.
function varargout = GenderRecogGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                             1、菜单栏
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1-1 一键完成：读入路径、提取特征、划分训练验证集、训练分类器
function MenuGetDataSets_Callback(hObject, eventdata, handles)
% hObject    handle to MenuGetDataSets (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
global classifierparam;
control.datasetpath = uigetdir('','选择原始数据集所在文件夹');
if ~ischar(control.datasetpath)
    warndlg('警告！您没有选择文件夹，请选择原始数据集所在文件夹','警告提示');
else
    % 在提取特征的过程中，整个面板应当不可操作
    allWidgetEnable(handles,0);
    hs = msgbox('正在训练','提示');
    ht = findobj(hs, 'Type', 'text');
    set(ht,'FontSize',8);
    set(hs, 'Resize', 'on');
    fprintf('正在提取特征...\n');
    %% 获取路径
    control.fold_list = dir(control.datasetpath);
    %% 特征提取
    category_num = length(control.fold_list)-2;
    % 获得信息构建结构体
    VOICES(category_num,1) = struct('name',[],'data',[],'num',[]);
    % 读取图像到结构体中
    fts_num = 39;       % 提取的特征数目(频域+mfcc+dmfcc)，此阶段所有特征都先提取出来再说
    for i=1:category_num
        VOICES(i).name = control.fold_list(i+2).name;
        % 每个类中的若干个文件
        file_list = dir([control.datasetpath,'\',VOICES(i).name]);
        VOICES(i).num = min(length(file_list)-2,30+round(5*rand()));   % 随机抽取部分数据
%         VOICES(i).num = length(file_list)-2;                            % 所有数据
        VOICES(i).data = zeros(VOICES(i).num,fts_num);
        for j=1:VOICES(i).num
            [currentvoice,fs] = audioread([control.datasetpath,'\',VOICES(i).name,'\',file_list(j+2).name]);
            feature = myfeature(currentvoice,fs);       % 提取各种特征
            VOICES(i).data(j,:) = feature(:)';          % 转为行向量存入结构体中
        end
        fprintf('第%d类/共%d类语音(共%d条数据)特征提取完毕\n',i,category_num,VOICES(i).num);
    end
    control.ftsstruct = VOICES;
    fprintf('特征已提取\n');
    %% 划分数据集
    category_num = size(control.ftsstruct,1);
    traindata = [];
    trainlabel = [];
    valdata = [];
    vallabel = [];
    for i=1:category_num
        file_num = VOICES(i).num;
        [val_index,train_index] = crossvalind('holdOut',file_num,control.TrainSegRatio);
        traindata = [traindata;VOICES(i).data(train_index,:)];
        trainlabel = [trainlabel;i*ones(sum(train_index),1)];
        valdata = [valdata;VOICES(i).data(val_index,:)];
        vallabel = [vallabel;i*ones(sum(val_index),1)];
    end
    control.traindata = traindata;
    control.trainlabel = trainlabel;
    control.valdata = valdata;
    control.vallabel = vallabel;
    fprintf('训练集/验证集划分完毕\n');
    %% 训练
    TraindataSelected = control.traindata;
    ValdataSelected = control.valdata;
    TraindataSelected(:,control.ftsCandidateChain==0) = [];
    ValdataSelected(:,control.ftsCandidateChain==0) = [];
    % 被筛选的原始信号也需要被保存，在NBtest时要用到
    classifierparam.TraindataSelected = TraindataSelected;
    classifierparam.ValdataSelected = ValdataSelected;
    % 开始训练分类器
    if strcmp(control.classifier,'DistCompare')==1
        % 训练测试
        Results = myDistDetermine(TraindataSelected,control.trainlabel,ValdataSelected);
        cfsmtx = mycfsmtx(control.vallabel,Results.predicted_label);
        % 数据保存
        classifierparam.DCtraincore = Results.DCtraincore;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'NaiveBayes')==1
        % 量化预处理
        ftsnum = size(TraindataSelected,2);     % 特征数值
        traindata_qualify = zeros(size(TraindataSelected));
        valdata_qualify = zeros(size(ValdataSelected));
        for i=1:ftsnum
            traindata_qualify(:,i) = mydiscretization(TraindataSelected(:,i),control.ftsRange);     % 特征数值量化阶ftsRange = 20;
            valdata_qualify(:,i) = mydiscretization(ValdataSelected(:,i),control.ftsRange);
        end
        % 训练测试
        [TrainingSets,ValidationSets] = myNaiveBayesTrain(traindata_qualify,control.trainlabel,...
            valdata_qualify,control.vallabel,control.ftsRange);
        predicted_label = myNaiveBayesValidation(TrainingSets,ValidationSets);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 数据保存
        classifierparam.NBTrainingSets = TrainingSets;
        classifierparam.NBValidationSets = ValidationSets;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'KMSKNN')==1
        % 聚类预处理
        train_class_num = mynumstatistic(control.trainlabel);
        train_female_num = train_class_num(1,2);
        train_male_num = train_class_num(2,2);
        [~,traindata_new_f,~,~,~] = mykmeans( TraindataSelected(1:train_female_num,:),control.clusters );       % clusters = 100;
        [~,traindata_new_m,~,~,~] = mykmeans( TraindataSelected(train_female_num+1:train_female_num+train_male_num,:),control.clusters );
        traindata_new = [traindata_new_f;traindata_new_m];
        trainlabel_new = [ones(control.clusters,1);2*ones(control.clusters,1)];
        % 训练测试
        KMSKNN_model = fitcknn(traindata_new,trainlabel_new,'NumNeighbors',control.Knearest); % Knearest=12
        predicted_label = KMSKNN_model.predict(ValdataSelected);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 数据保存
        classifierparam.KMSKNNmodel = KMSKNN_model;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'KNN')==1
        % 训练测试
        KNN_model = fitcknn(TraindataSelected,control.trainlabel,'NumNeighbors',7); 
        predicted_label = KNN_model.predict(ValdataSelected);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 数据保存
        classifierparam.KNNmodel = KNN_model;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'SVM')==1
        % 寻优预处理(粗寻优+精寻优)
        [cmin,cmax,gmin,gmax,v,cstep,gstep,accstep] = deal(-15,15,-15,15,3,2,2,2);
        [~,c_temp,g_temp] = SVMcgForClass(control.trainlabel,TraindataSelected,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
        [cmin,cmax,gmin,gmax,v,cstep,gstep,accstep] = deal(log(c_temp)/log(2)-2,log(c_temp)/log(2)+2,...
            log(g_temp)/log(2)-2,log(g_temp)/log(2)+2,3,0.25,0.25,0.5);
        [~,bestc,bestg] = SVMcgForClass(control.trainlabel,TraindataSelected,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
        % 训练测试
        cmd = [' -s ',num2str(0),' -c ',num2str(bestc),' -g ',num2str(bestg)];
        svm_model = svmtrain(control.trainlabel,TraindataSelected,cmd);
        [predicted_label, ~, ~] = svmpredict(control.vallabel,ValdataSelected,svm_model);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 数据保存
        classifierparam.SVMmodel = svm_model;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    else
        errordlg('错误！该分类器不存在','错误提示');
    end
    % 训练分类器完成，恢复面板的可操作性
    allWidgetEnable(handles,1);
    % 提示信息
    fprintf('训练完毕\n');
    hs = msgbox('训练完毕','提示');
    ht = findobj(hs, 'Type', 'text');
    set(ht,'FontSize',8);
    set(hs, 'Resize', 'on');
    
    %% 训练阶段，复选框后面的静态文本框（中间）一律不显示特征数据
    % 1 text_meanfreq
    set(handles.text_meanfreq,'string','meanfreq');
    % 2 text_freqstd
    set(handles.text_freqstd,'string','freqstd');
    % 3 text_meanfreq
    set(handles.text_meanfreq,'string','meanfreq');
    % 4 text_Q25
    set(handles.text_Q25,'string','Q25');
    % 5 text_Q75
    set(handles.text_Q75,'string','Q75');
    % 6 text_IRQ
    set(handles.text_IRQ,'string','IRQ');
    % 7 text_skew
    set(handles.text_skew,'string','skew');
    % 8 text_kurt
    set(handles.text_kurt,'string','kurt');
    % 9 text_spent
    set(handles.text_spent,'string','spent');
    % 10 text_mode
    set(handles.text_mode,'string','mode');
    % 11 text_meanpitch
    set(handles.text_meanpitch,'string','meanpitch');
    % 12 text_maxpitch
    set(handles.text_maxpitch,'string','maxpitch');
    % 13 text_minpitch
    set(handles.text_minpitch,'string','minpitch');  
    
    %% 训练阶段，复选框后面的静态文本框（上下）显示男女声在此项上的均值
    TraindataSelected4male = TraindataSelected;
    TraindataSelected4male(control.trainlabel==1,:) = [];
    TraindataSelected4male = mean(TraindataSelected4male);
    TraindataSelected4female = TraindataSelected;
    TraindataSelected4female(control.trainlabel==2,:) = [];
    TraindataSelected4female = mean(TraindataSelected4female);
    k = 1;
    % 1 text_meanfreq
    if control.ftsCandidateChain(1)==1
        set(handles.textM1,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF1,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM1,'string','——');
        set(handles.textF1,'string','——'); 
    end
    % 2 text_freqstd
    if control.ftsCandidateChain(2)==1
        set(handles.textM2,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF2,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM2,'string','——');
        set(handles.textF2,'string','——'); 
    end
    % 3 text_meanfreq
    if control.ftsCandidateChain(3)==1
        set(handles.textM3,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF3,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM3,'string','——');
        set(handles.textF3,'string','——'); 
    end
    % 4 text_Q25
    if control.ftsCandidateChain(4)==1
        set(handles.textM4,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF4,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM4,'string','——');
        set(handles.textF4,'string','——'); 
    end
    % 5 text_Q75
    if control.ftsCandidateChain(5)==1
        set(handles.textM5,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF5,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM5,'string','——');
        set(handles.textF5,'string','——'); 
    end
    % 6 text_IRQ
    if control.ftsCandidateChain(6)==1
        set(handles.textM6,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF6,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM6,'string','——');
        set(handles.textF6,'string','——'); 
    end
    % 7 text_skew
    if control.ftsCandidateChain(7)==1
        set(handles.textM7,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF7,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM7,'string','——');
        set(handles.textF7,'string','——');
    end
    % 8 text_kurt
    if control.ftsCandidateChain(8)==1
        set(handles.textM8,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF8,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM8,'string','——');
        set(handles.textF8,'string','——'); 
    end
    % 9 text_spent
    if control.ftsCandidateChain(9)==1
        set(handles.textM9,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF9,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM9,'string','——');
        set(handles.textF9,'string','——'); 
    end
    % 10 text_mode
    if control.ftsCandidateChain(10)==1
        set(handles.textM10,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF10,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM10,'string','——');
        set(handles.textF10,'string','——'); 
    end
    % 11 text_meanpitch
    if control.ftsCandidateChain(11)==1
        set(handles.textM11,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF11,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM11,'string','——');
        set(handles.textF11,'string','——'); 
    end
    % 12 text_maxpitch
    if control.ftsCandidateChain(12)==1
        set(handles.textM12,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF12,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM12,'string','——');
        set(handles.textF12,'string','——'); 
    end
    % 13 text_minpitch
    if control.ftsCandidateChain(13)==1
        set(handles.textM13,'string',num2str(TraindataSelected4male(k),'%.3f'));
        set(handles.textF13,'string',num2str(TraindataSelected4female(k),'%.3f'));   
        k = k+1;
    else
        set(handles.textM13,'string','——');
        set(handles.textF13,'string','——');
    end 
    % 自检
    if k~=sum(control.ftsCandidateChain(1:13))+1
        errordlg('错误！特征数量与显示数量不符','错误提示');
    end
    %% 标志位
    % 训练结束后，所有特征改变标志位清零
    control.featurechanged = zeros(1,15);
    % 训练结束后，选择的划分比例由 current 状态变为 past（当前变成上一轮）
    control.PastSplitRatioChoice = control.CurrentSplitRatioChoice;
    control.PastClassifierChoice = control.CurrentClassifierChoice;
end

% 1-2 读入语音--------------------------------------------------------------------
function MenuGetVoice_Callback(hObject, eventdata, handles)
% hObject    handle to MenuGetVoice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
[filename,pathname] = uigetfile({'*.wav';'*.mp3';'*.ogg';'*.au';'*.flac'},'Select a voice file');
[x,fs] = audioread([pathname,filename]);
recorder.voice = x;
param.fs = fs;
% 绘图
axes(handles.axes_time);
t = (0:length(x)-1)/fs;
plot(t,x);xlabel('Time/s');ylabel('Amplitude');title('Speech waveform');
% 提示信息
hs = msgbox('语音数据已读取完毕','提示');
ht = findobj(hs, 'Type', 'text');     
set(ht,'FontSize',8);     
set(hs, 'Resize', 'on'); 

% 3-1-1 选择训练集比例为50% -------------------------------------------
function MenuFiftyFifty_Callback(hObject, eventdata, handles)
% hObject    handle to MenuFiftyFifty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.TrainSegRatio = 0.5;
set(handles.MenuFiftyFifty,'Checked','on');     % 前面打勾
set(handles.MenuSixtyForty,'Checked','off');
set(handles.MenuSeventyThirty,'Checked','off');
set(handles.MenuEightyTwenty,'Checked','off');
set(handles.MenuNintyTen,'Checked','off');
control.CurrentSplitRatioChoice = 1;


% 3-1-2 选择训练集比例为60% -------------------------------------------
function MenuSixtyForty_Callback(hObject, eventdata, handles)
% hObject    handle to MenuSixtyForty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.TrainSegRatio = 0.6;
set(handles.MenuFiftyFifty,'Checked','off');    
set(handles.MenuSixtyForty,'Checked','on');     % 前面打勾
set(handles.MenuSeventyThirty,'Checked','off');
set(handles.MenuEightyTwenty,'Checked','off');
set(handles.MenuNintyTen,'Checked','off');
control.CurrentSplitRatioChoice = 2;

% 3-1-3 选择训练集比例为70% ------------------------------------------
function MenuSeventyThirty_Callback(hObject, eventdata, handles)
% hObject    handle to MenuSeventyThirty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.TrainSegRatio = 0.7;
set(handles.MenuFiftyFifty,'Checked','off');     
set(handles.MenuSixtyForty,'Checked','off');
set(handles.MenuSeventyThirty,'Checked','on');      % 前面打勾
set(handles.MenuEightyTwenty,'Checked','off');
set(handles.MenuNintyTen,'Checked','off');
control.CurrentSplitRatioChoice = 3;

% 3-1-4 选择训练集比例为80% ------------------------------------------
function MenuEightyTwenty_Callback(hObject, eventdata, handles)
% hObject    handle to MenuEightyTwenty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.TrainSegRatio = 0.8;
set(handles.MenuFiftyFifty,'Checked','off');     
set(handles.MenuSixtyForty,'Checked','off');
set(handles.MenuSeventyThirty,'Checked','off');
set(handles.MenuEightyTwenty,'Checked','on');       % 前面打勾
set(handles.MenuNintyTen,'Checked','off');
control.CurrentSplitRatioChoice = 4;

% 3-1-5 选择训练集比例为90% -------------------------------------------
function MenuNintyTen_Callback(hObject, eventdata, handles)
% hObject    handle to MenuNintyTen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.TrainSegRatio = 0.9;
set(handles.MenuFiftyFifty,'Checked','off');
set(handles.MenuSixtyForty,'Checked','off');
set(handles.MenuSeventyThirty,'Checked','off');
set(handles.MenuEightyTwenty,'Checked','off');
set(handles.MenuNintyTen,'Checked','on');       % 前面打勾
control.CurrentSplitRatioChoice = 5;

% 3-2-1 选择分类器为 DistCompare --------------------------------------
function MenuDistCompare_Callback(hObject, eventdata, handles)
% hObject    handle to MenuDistCompare (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.classifier = 'DistCompare';
set(handles.MenuDistCompare,'Checked','on');     % 前面打勾
set(handles.MenuNaiveBayes,'Checked','off');
set(handles.MenuKMSKNN,'Checked','off');
set(handles.MenuKNN,'Checked','off');
set(handles.MenuSVM,'Checked','off');      
control.CurrentClassifierChoice = 1;                  % 默认分类器 DistCompare

% 3-2-2 选择分类器为 NaiveBayes ---------------------------------------
function MenuNaiveBayes_Callback(hObject, eventdata, handles)
% hObject    handle to MenuNaiveBayes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.classifier = 'NaiveBayes';
set(handles.MenuDistCompare,'Checked','off');
set(handles.MenuNaiveBayes,'Checked','on');    % 前面打勾
set(handles.MenuKMSKNN,'Checked','off');
set(handles.MenuKNN,'Checked','off');
set(handles.MenuSVM,'Checked','off');       
control.CurrentClassifierChoice = 2;

% 3-2-3 选择分类器为 KMSKNN --------------------------------------------
function MenuKMSKNN_Callback(hObject, eventdata, handles)
% hObject    handle to MenuKMSKNN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.classifier = 'KMSKNN';
set(handles.MenuDistCompare,'Checked','off');
set(handles.MenuNaiveBayes,'Checked','off');
set(handles.MenuKMSKNN,'Checked','on');     % 前面打勾
set(handles.MenuKNN,'Checked','off');
set(handles.MenuSVM,'Checked','off');       
control.CurrentClassifierChoice = 3;

% 3-2-4 选择分类器为 KNN ---------------------------------------------
function MenuKNN_Callback(hObject, eventdata, handles)
% hObject    handle to MenuKNN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.classifier = 'KNN';
set(handles.MenuDistCompare,'Checked','off');
set(handles.MenuNaiveBayes,'Checked','off');
set(handles.MenuKMSKNN,'Checked','off');
set(handles.MenuKNN,'Checked','on');        % 前面打勾
set(handles.MenuSVM,'Checked','off');
control.CurrentClassifierChoice = 4;

% 3-2-5 选择分类器为 SVM --------------------------------------------
function MenuSVM_Callback(hObject, eventdata, handles)
% hObject    handle to MenuSVM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global control;
control.classifier = 'SVM';
set(handles.MenuDistCompare,'Checked','off');
set(handles.MenuNaiveBayes,'Checked','off');
set(handles.MenuKMSKNN,'Checked','off');
set(handles.MenuKNN,'Checked','off');
set(handles.MenuSVM,'Checked','on');       % 前面打勾
control.CurrentClassifierChoice = 5;

% 4-1 软件使用帮助------------------------------------------------------
function MenuGuide_Callback(hObject, eventdata, handles)
% hObject    handle to MenuGuide (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hs = msgbox({'软件使用帮助:';...
            '这是一款简易的语音性别识别软件，具有如下功能：';'';...
            '1. （作者较懒，用户自己体验吧）；';...
            '2. ...；';...
            '3. ...；';...
            '4. ...；';...
            '4. ...。';'';...
            '其中，...：';...
            '        ...';'        ...';'        ...';...
            '        ...';'';...
            '软件还...，包括：';
            '        ...';...
            '        ...';...
            '        ...';...
            '        ...';...
            '        ...';...
            '        ...';...
            },'UserGuide');
%改变字体大小
ht = findobj(hs, 'Type', 'text');
set(ht,'FontSize',8);
%改变对话框大小
set(hs, 'Resize', 'on'); 

% 4-2 软件版本说明-----------------------------------------------------
function MenuVersion_Callback(hObject, eventdata, handles)
% hObject    handle to MenuVersion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hs = msgbox({'软件版本说明:';'';'Version: 0.3 ';'';...
             'Author: Chen Tianyang';'';...
             'Data:2019-03-26';''},'Version Information');
%改变字体大小
ht = findobj(hs, 'Type', 'text');
set(ht,'FontSize',8);
%改变对话框大小
set(hs, 'Resize', 'on'); 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                             2、工具栏
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------------------------------------------------
function showgrid_OnCallback(hObject, eventdata, handles)
% hObject    handle to showgrid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes_time);
grid on;

% --------------------------------------------------------------------
function showgrid_OffCallback(hObject, eventdata, handles)
% hObject    handle to showgrid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes_time);
grid off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                        3、 Classifier Properties                             
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         4、Voice Recognition                             
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 开始/录音按钮 --------------------------------------------------------
function StartRecord_Callback(hObject, eventdata, handles)
% hObject    handle to StartRecord (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
% 录音中，面板不可操作
allWidgetEnable(handles,0);
if recorder.begin == false
    recorder.begin = true;
    recorder.R = audiorecorder(param.fs,16,1);
    record(recorder.R); %开始录音
    set(handles.StartRecord,'string','停止');
    set(handles.StartRecord,'enable','on');
    return;
else
    recorder.begin = false;
    stop(recorder.R); 
    recorder.voice = getaudiodata(recorder.R);
    index = param.startindex+param.number;
    folder = [pwd,'\recorder'];
    if ~isdir(folder)
        mkdir(folder);
    end
    filename = [folder,'\',param.label,'_',num2str(index),'.wav'];
    audiowrite(filename,recorder.voice,param.fs);
    param.number = param.number+1;       
    set(handles.StartRecord,'string','录音');
    % 绘图
    axes(handles.axes_time);
    t = (0:length(recorder.voice)-1)/param.fs;
    plot(t,recorder.voice);xlabel('Time/s');ylabel('Amplitude');title('Speech waveform');
end
% 录音完毕，面板恢复正常操作
allWidgetEnable(handles,1);
% 显示栏的结论和数据清除
set(handles.textCurrentGender,'String','性别');
% 1 text_meanfreq
set(handles.text_meanfreq,'string','meanfreq');
% 2 text_freqstd
set(handles.text_freqstd,'string','freqstd');
% 3 text_meanfreq
set(handles.text_meanfreq,'string','meanfreq');
% 4 text_Q25
set(handles.text_Q25,'string','Q25');
% 5 text_Q75
set(handles.text_Q75,'string','Q75');
% 6 text_IRQ
set(handles.text_IRQ,'string','IRQ');
% 7 text_skew
set(handles.text_skew,'string','skew');
% 8 text_kurt
set(handles.text_kurt,'string','kurt');
% 9 text_spent
set(handles.text_spent,'string','spent');
% 10 text_mode
set(handles.text_mode,'string','mode');
% 11 text_meanpitch
set(handles.text_meanpitch,'string','meanpitch');
% 12 text_maxpitch
set(handles.text_maxpitch,'string','maxpitch');
% 13 text_minpitch
set(handles.text_minpitch,'string','minpitch');  

% 2 从本地加载语音信号按钮 --------------------------------------------------
function ChooseRecord_Callback(hObject, eventdata, handles)
% hObject    handle to ChooseRecord (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
[filename,pathname] = uigetfile({'*.wav';'*.mp3';'*.ogg';'*.au';'*.flac'},'Select a voice file');
[x,fs] = audioread([pathname,filename]);
recorder.voice = x;
param.fs = fs;
% 绘图
axes(handles.axes_time);
t = (0:length(x)-1)/fs;
plot(t,x);xlabel('Time/s');ylabel('Amplitude');title('Speech waveform');
% 显示栏的结论和数据清除
set(handles.textCurrentGender,'String','性别');
% 1 text_meanfreq
set(handles.text_meanfreq,'string','meanfreq');
% 2 text_freqstd
set(handles.text_freqstd,'string','freqstd');
% 3 text_meanfreq
set(handles.text_meanfreq,'string','meanfreq');
% 4 text_Q25
set(handles.text_Q25,'string','Q25');
% 5 text_Q75
set(handles.text_Q75,'string','Q75');
% 6 text_IRQ
set(handles.text_IRQ,'string','IRQ');
% 7 text_skew
set(handles.text_skew,'string','skew');
% 8 text_kurt
set(handles.text_kurt,'string','kurt');
% 9 text_spent
set(handles.text_spent,'string','spent');
% 10 text_mode
set(handles.text_mode,'string','mode');
% 11 text_meanpitch
set(handles.text_meanpitch,'string','meanpitch');
% 12 text_maxpitch
set(handles.text_maxpitch,'string','maxpitch');
% 13 text_minpitch
set(handles.text_minpitch,'string','minpitch');  

% 3 识别该信号是男声还是女声 ----------------------------------------------
function pushbuttonGoJudge_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonGoJudge (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
global control;
global classifierparam;
if isempty(recorder.voice)
    errordlg('错误！没有发现语音，请先录音或读取本地语音','错误提示');
elseif sum(control.ftsCandidateChain)==0
    errordlg('错误！至少应选择一种语音特征，否则无法判断性别','错误提示');
elseif sum(control.featurechanged)~=0
    errordlg('错误！特征组合已改变，必须重新训练','错误提示');
elseif control.CurrentSplitRatioChoice ~= control.PastSplitRatioChoice
    errordlg('错误！训练集验证集比例已改变，必须重新训练','错误提示');
elseif control.CurrentClassifierChoice ~= control.PastClassifierChoice
    errordlg('错误！分类器已改变，必须重新训练','错误提示');
else
    % 提取各种特征
    feature = myfeature(recorder.voice,param.fs);
    feature = feature(:)';              % 特征转为行向量
    % 特征筛选
    SelectedFeature = feature;          
    SelectedFeature(control.ftsCandidateChain==0) = [];
    % 判断测试的特征和训练/验证的特征是否一样
    % 判断当前采用的是哪种分类器
    if strcmp(control.classifier,'DistCompare')==1
        if isempty(classifierparam.DCtraincore)
            errordlg('错误！Dist Comapre 分类器不存在,请先训练此分类器','错误提示');
        else
            % DistCompare 检测
            predicted_label = myDistDetermineTest(classifierparam.DCtraincore,SelectedFeature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','female');
            else
                set(handles.textCurrentGender,'string','male');
            end
        end
    elseif strcmp(control.classifier,'NaiveBayes')==1
        if isempty(classifierparam.NBTrainingSets)||isempty(classifierparam.NBValidationSets)
            errordlg('错误！Naive Bayes 分类器不存在,请先训练此分类器','错误提示');
        else 
            % 首先从 NBTrainingSets 中得出每个特征的最大最小值
            feature_max = max([classifierparam.TraindataSelected;classifierparam.ValdataSelected]);
            feature_min = min([classifierparam.TraindataSelected;classifierparam.ValdataSelected]);
            % 以此为依据对新特征进行增量量化
            feature_qualify = mydiscretization2(feature_max,feature_min,control.ftsRange,SelectedFeature);
            % NaiveBayes 检测
            predicted_label = myNaiveBayesTest(classifierparam.NBTrainingSets,feature_qualify);
            if predicted_label==1
                set(handles.textCurrentGender,'string','female');
            else
                set(handles.textCurrentGender,'string','male');
            end
        end
    elseif strcmp(control.classifier,'KMSKNN')==1
        if isempty(classifierparam.KMSKNNmodel)
            errordlg('错误！KMSKNN 分类器不存在,请先训练此分类器','错误提示');
        else
            KMSKNN_model = classifierparam.KMSKNNmodel;
            predicted_label = KMSKNN_model.predict(SelectedFeature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','female');
            else
                set(handles.textCurrentGender,'string','male');
            end
        end
    elseif strcmp(control.classifier,'KNN')==1
        if isempty(classifierparam.KNNmodel)
            errordlg('错误！KNN 分类器不存在,请先训练此分类器','错误提示');
        else
            KNN_model = classifierparam.KNNmodel;
            predicted_label = KNN_model.predict(SelectedFeature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','female');
            else
                set(handles.textCurrentGender,'string','male');
            end
        end
    elseif strcmp(control.classifier,'SVM')==1
        if isempty(classifierparam.SVMmodel)
            errordlg('错误！SVM 分类器不存在,请先训练此分类器','错误提示');
        else
            % 测试
            [predicted_label, ~, ~] = svmpredict(1,SelectedFeature,classifierparam.SVMmodel);   % 第一个参数"1"是待判断数据的伪标签
            if predicted_label==1
                set(handles.textCurrentGender,'string','female');
            else
                set(handles.textCurrentGender,'string','male');
            end
        end
    else
        errordlg('错误！该分类器不存在','错误提示');
    end
    % 在复选框后的静态文本框内显示这些特征的数值
    % 1 text_meanfreq
    if control.ftsCandidateChain(1)==0
        set(handles.text_meanfreq,'string','——');
    else
        set(handles.text_meanfreq,'string',num2str(feature(1),'%.3f'));
    end
    % 2 text_freqstd
    if control.ftsCandidateChain(2)==0
        set(handles.text_freqstd,'string','——');
    else
        set(handles.text_freqstd,'string',num2str(feature(2),'%.3f'));
    end
    % 3 text_meanfreq
    if control.ftsCandidateChain(3)==0
        set(handles.text_midfreq,'string','——');
    else
        set(handles.text_midfreq,'string',num2str(feature(3),'%.3f'));
    end
    % 4 text_Q25
    if control.ftsCandidateChain(4)==0
        set(handles.text_Q25,'string','——');
    else
        set(handles.text_Q25,'string',num2str(feature(4),'%.3f'));
    end
    % 5 text_Q75
    if control.ftsCandidateChain(5)==0
        set(handles.text_Q75,'string','——');
    else
        set(handles.text_Q75,'string',num2str(feature(5),'%.3f'));
    end
    % 6 text_IRQ
    if control.ftsCandidateChain(6)==0
        set(handles.text_IRQ,'string','——');
    else
        set(handles.text_IRQ,'string',num2str(feature(6),'%.3f'));
    end
    % 7 text_skew
    if control.ftsCandidateChain(7)==0
        set(handles.text_skew,'string','——');
    else
        set(handles.text_skew,'string',num2str(feature(7),'%.3f'));
    end
    % 8 text_kurt
    if control.ftsCandidateChain(8)==0
        set(handles.text_kurt,'string','——');
    else
        set(handles.text_kurt,'string',num2str(feature(8),'%.3f'));
    end
    % 9 text_spent
    if control.ftsCandidateChain(9)==0
        set(handles.text_spent,'string','——');
    else
        set(handles.text_spent,'string',num2str(feature(9),'%.3f'));
    end
    % 10 text_mode
    if control.ftsCandidateChain(10)==0
        set(handles.text_mode,'string','——');
    else
        set(handles.text_mode,'string',num2str(feature(10),'%.3f'));
    end
    % 11 text_meanpitch
    if control.ftsCandidateChain(11)==0
        set(handles.text_meanpitch,'string','——');
    else
        set(handles.text_meanpitch,'string',num2str(feature(11),'%.3f'));
    end
    % 12 text_maxpitch
    if control.ftsCandidateChain(12)==0
        set(handles.text_maxpitch,'string','——');
    else
        set(handles.text_maxpitch,'string',num2str(feature(12),'%.3f'));
    end
    % 13 text_minpitch
    if control.ftsCandidateChain(13)==0
        set(handles.text_minpitch,'string','——');
    else
        set(handles.text_minpitch,'string',num2str(feature(13),'%.3f'));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         5、Axes Related
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes on button press in pushbuttonPlay.
function pushbuttonPlay_Callback(hObject, eventdata, handles)
% hObject    handle to pushbuttonPlay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
if ~isempty(recorder.voice)&&~isempty(param.fs)
    sound(recorder.voice,param.fs);
end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         6、Feature Choice
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- 1 meanfreq
function checkbox_meanfreq_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_meanfreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_meanfreq
global control;
if control.featurechanged(1)==0
    control.featurechanged(1) = 1;
else
    control.featurechanged(1) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(1) = 1;
else
    control.ftsCandidateChain(1) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end


% --- 2 freqstd
function checkbox_freqstd_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_freqstd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_freqstd
global control;
if control.featurechanged(2)==0
    control.featurechanged(2) = 1;
else
    control.featurechanged(2) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(2) = 1;
else
    control.ftsCandidateChain(2) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 3 midfreq.
function checkbox_midfreq_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_midfreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_midfreq
global control;
if control.featurechanged(3)==0
    control.featurechanged(3) = 1;
else
    control.featurechanged(3) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(3) = 1;
else
    control.ftsCandidateChain(3) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 4 Q25.
function checkbox_Q25_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_Q25 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_Q25
global control;
if control.featurechanged(4)==0
    control.featurechanged(4) = 1;
else
    control.featurechanged(4) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(4) = 1;
else
    control.ftsCandidateChain(4) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 5 Q75.
function checkbox_Q75_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_Q75 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_Q75
global control;
if control.featurechanged(5)==0
    control.featurechanged(5) = 1;
else
    control.featurechanged(5) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(5) = 1;
else
    control.ftsCandidateChain(5) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 6 IRQ.
function checkbox_IRQ_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_IRQ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_IRQ
global control;
if control.featurechanged(6)==0
    control.featurechanged(6) = 1;
else
    control.featurechanged(6) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(6) = 1;
else
    control.ftsCandidateChain(6) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 7 skew.
function checkbox_skew_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_skew (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_skew
global control;
if control.featurechanged(7)==0
    control.featurechanged(7) = 1;
else
    control.featurechanged(7) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(7) = 1;
else
    control.ftsCandidateChain(7) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 8 kurt.
function checkbox_kurt_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_kurt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_kurt
global control;
if control.featurechanged(8)==0
    control.featurechanged(8) = 1;
else
    control.featurechanged(8) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(8) = 1;
else
    control.ftsCandidateChain(8) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 9 spent.
function checkbox_spent_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_spent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_spent
global control;
if control.featurechanged(9)==0
    control.featurechanged(9) = 1;
else
    control.featurechanged(9) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(9) = 1;
else
    control.ftsCandidateChain(9) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 10 mode.
function checkbox_mode_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_mode (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_mode
global control;
if control.featurechanged(10)==0
    control.featurechanged(10) = 1;
else
    control.featurechanged(10) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(10) = 1;
else
    control.ftsCandidateChain(10) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 11 meanpitch.
function checkbox_meanpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_meanpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_meanpitch
global control;
if control.featurechanged(11)==0
    control.featurechanged(11) = 1;
else
    control.featurechanged(11) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(11) = 1;
else
    control.ftsCandidateChain(11) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 12 maxpitch.
function checkbox_maxpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_maxpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_maxpitch
global control;
if control.featurechanged(12)==0
    control.featurechanged(12) = 1;
else
    control.featurechanged(12) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(12) = 1;
else
    control.ftsCandidateChain(12) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 13 minpitch.
function checkbox_minpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_minpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_minpitch
global control;
if control.featurechanged(13)==0
    control.featurechanged(13) = 1;
else
    control.featurechanged(13) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(13) = 1;
else
    control.ftsCandidateChain(13) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end


% --- 14 mfcc.
function checkbox_mfcc_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_mfcc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_mfcc
global control;
if control.featurechanged(14)==0
    control.featurechanged(14) = 1;
else
    control.featurechanged(14) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(14:26) = 1;
else
    control.ftsCandidateChain(14:26) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 15 dmfcc.
function checkbox_dmfcc_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_dmfcc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_dmfcc
global control;
if control.featurechanged(15)==0
    control.featurechanged(15) = 1;
else
    control.featurechanged(15) = 0;
end
if get(hObject,'Value')==1
    control.ftsCandidateChain(27:39) = 1;
else
    control.ftsCandidateChain(27:39) = 0;
end
if sum(control.ftsCandidateChain)~=39
    set(handles.checkbox_chooseall,'Value',0);
end

% --- 16 dmfcc chooseall
function checkbox_chooseall_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_chooseall (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_chooseall
global control;
if get(hObject,'Value')==1
    set(handles.checkbox_meanfreq,'Value',1);
    set(handles.checkbox_freqstd,'Value',1);
    set(handles.checkbox_midfreq,'Value',1);
    set(handles.checkbox_Q25,'Value',1);
    set(handles.checkbox_Q75,'Value',1);
    set(handles.checkbox_IRQ,'Value',1);
    set(handles.checkbox_skew,'Value',1);
    set(handles.checkbox_kurt,'Value',1);
    set(handles.checkbox_spent,'Value',1);
    set(handles.checkbox_mode,'Value',1);
    set(handles.checkbox_meanpitch,'Value',1);
    set(handles.checkbox_maxpitch,'Value',1);
    set(handles.checkbox_minpitch,'Value',1);
    set(handles.checkbox_mfcc,'Value',1);
    set(handles.checkbox_dmfcc,'Value',1);
    control.ftsCandidateChain(1:39) = 1;
else
    set(handles.checkbox_meanfreq,'Value',0);
    set(handles.checkbox_freqstd,'Value',0);
    set(handles.checkbox_midfreq,'Value',0);
    set(handles.checkbox_Q25,'Value',0);
    set(handles.checkbox_Q75,'Value',0);
    set(handles.checkbox_IRQ,'Value',0);
    set(handles.checkbox_skew,'Value',0);
    set(handles.checkbox_kurt,'Value',0);
    set(handles.checkbox_spent,'Value',0);
    set(handles.checkbox_mode,'Value',0);
    set(handles.checkbox_meanpitch,'Value',0);
    set(handles.checkbox_maxpitch,'Value',0);
    set(handles.checkbox_minpitch,'Value',0);
    set(handles.checkbox_mfcc,'Value',0);
    set(handles.checkbox_dmfcc,'Value',0);
    control.ftsCandidateChain(1:39) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         7、扫尾函数
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
rmpath(genpath([pwd,'\myfunctions']));
rmpath(genpath([pwd,'\libsvm322']));
disp('Exit GendeRecogGUI');
delete(hObject);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         8、其他辅助被调函数
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function allWidgetEnable(handles,enable)
if enable == 1
    set(handles.textCurrentAccuracy,'enable','on');
    set(handles.textCurrentRecallRate,'enable','on');
    set(handles.StartRecord,'enable','on');
    set(handles.ChooseRecord,'enable','on');
    set(handles.pushbuttonGoJudge,'enable','on');
    set(handles.pushbuttonPlay,'enable','on');
    set(handles.textCurrentGender,'enable','on');
    set(handles.checkbox_meanfreq,'enable','on');
    set(handles.checkbox_freqstd,'enable','on');
    set(handles.checkbox_midfreq,'enable','on');
    set(handles.checkbox_Q25,'enable','on');
    set(handles.checkbox_Q75,'enable','on');
    set(handles.checkbox_IRQ,'enable','on');
    set(handles.checkbox_skew,'enable','on');
    set(handles.checkbox_kurt,'enable','on');
    set(handles.checkbox_spent,'enable','on');
    set(handles.checkbox_mode,'enable','on');
    set(handles.checkbox_meanpitch,'enable','on');
    set(handles.checkbox_maxpitch,'enable','on');
    set(handles.checkbox_minpitch,'enable','on');
    set(handles.checkbox_mfcc,'enable','on');
    set(handles.checkbox_dmfcc,'enable','on');
    set(handles.checkbox_chooseall,'enable','on');
    
    set(handles.text_meanfreq,'enable','on');
    set(handles.text_freqstd,'enable','on');
    set(handles.text_midfreq,'enable','on');
    set(handles.text_Q25,'enable','on');
    set(handles.text_Q75,'enable','on');
    set(handles.text_IRQ,'enable','on');
    set(handles.text_skew,'enable','on');
    set(handles.text_kurt,'enable','on');
    set(handles.text_spent,'enable','on');
    set(handles.text_mode,'enable','on');
    set(handles.text_meanpitch,'enable','on');
    set(handles.text_maxpitch,'enable','on');
    set(handles.text_minpitch,'enable','on');
    
    set(handles.RecordTip,'enable','on');
    set(handles.textM1,'enable','on');
    set(handles.textM2,'enable','on');
    set(handles.textM3,'enable','on');
    set(handles.textM4,'enable','on');
    set(handles.textM5,'enable','on');
    set(handles.textM6,'enable','on');
    set(handles.textM7,'enable','on');
    set(handles.textM8,'enable','on');
    set(handles.textM9,'enable','on');
    set(handles.textM10,'enable','on');
    set(handles.textM11,'enable','on');
    set(handles.textM12,'enable','on');
    set(handles.textM13,'enable','on');
    set(handles.textF1,'enable','on');
    set(handles.textF2,'enable','on');
    set(handles.textF3,'enable','on');
    set(handles.textF4,'enable','on');
    set(handles.textF5,'enable','on');
    set(handles.textF6,'enable','on');
    set(handles.textF7,'enable','on');
    set(handles.textF8,'enable','on');
    set(handles.textF9,'enable','on');
    set(handles.textF10,'enable','on');
    set(handles.textF11,'enable','on');
    set(handles.textF12,'enable','on');
    set(handles.textF13,'enable','on');
    
    set(handles.textmeanmale1,'enable','on');
    set(handles.textmeanmale2,'enable','on');
    set(handles.textmeanmale3,'enable','on');
    set(handles.textmeanmale4,'enable','on');
    set(handles.textmeanmale5,'enable','on');
    set(handles.textmeanmale6,'enable','on');
    set(handles.textmeanmale7,'enable','on');
    set(handles.textmeanmale8,'enable','on');
    set(handles.textmeanmale9,'enable','on');
    set(handles.textmeanmale10,'enable','on');
    set(handles.textmeanmale11,'enable','on');
    set(handles.textmeanmale12,'enable','on');
    set(handles.textmeanmale13,'enable','on');
    set(handles.textmeanfemale1,'enable','on');
    set(handles.textmeanfemale2,'enable','on');
    set(handles.textmeanfemale3,'enable','on');
    set(handles.textmeanfemale4,'enable','on');
    set(handles.textmeanfemale5,'enable','on');
    set(handles.textmeanfemale6,'enable','on');
    set(handles.textmeanfemale7,'enable','on');
    set(handles.textmeanfemale8,'enable','on');
    set(handles.textmeanfemale9,'enable','on');
    set(handles.textmeanfemale10,'enable','on');
    set(handles.textmeanfemale11,'enable','on');
    set(handles.textmeanfemale12,'enable','on');
    set(handles.textmeanfemale13,'enable','on');
    
    set(handles.textcurrent1,'enable','on');
    set(handles.textcurrent2,'enable','on');
    set(handles.textcurrent3,'enable','on');
    set(handles.textcurrent4,'enable','on');
    set(handles.textcurrent5,'enable','on');
    set(handles.textcurrent6,'enable','on');
    set(handles.textcurrent7,'enable','on');
    set(handles.textcurrent8,'enable','on');
    set(handles.textcurrent9,'enable','on');
    set(handles.textcurrent10,'enable','on');
    set(handles.textcurrent11,'enable','on');
    set(handles.textcurrent12,'enable','on');
    set(handles.textcurrent13,'enable','on');
    

else
    set(handles.textCurrentAccuracy,'enable','off');
    set(handles.textCurrentRecallRate,'enable','off');
    set(handles.StartRecord,'enable','off');
    set(handles.ChooseRecord,'enable','off');
    set(handles.pushbuttonGoJudge,'enable','off');
    set(handles.pushbuttonPlay,'enable','off');
    set(handles.textCurrentGender,'enable','off');
    set(handles.checkbox_meanfreq,'enable','off');
    set(handles.checkbox_freqstd,'enable','off');
    set(handles.checkbox_midfreq,'enable','off');
    set(handles.checkbox_Q25,'enable','off');
    set(handles.checkbox_Q75,'enable','off');
    set(handles.checkbox_IRQ,'enable','off');
    set(handles.checkbox_skew,'enable','off');
    set(handles.checkbox_kurt,'enable','off');
    set(handles.checkbox_spent,'enable','off');
    set(handles.checkbox_mode,'enable','off');
    set(handles.checkbox_meanpitch,'enable','off');
    set(handles.checkbox_maxpitch,'enable','off');
    set(handles.checkbox_minpitch,'enable','off');
    set(handles.checkbox_mfcc,'enable','off');
    set(handles.checkbox_dmfcc,'enable','off');
    set(handles.checkbox_chooseall,'enable','off');
    set(handles.text_meanfreq,'enable','off');
    set(handles.text_freqstd,'enable','off');
    set(handles.text_midfreq,'enable','off');
    set(handles.text_Q25,'enable','off');
    set(handles.text_Q75,'enable','off');
    set(handles.text_IRQ,'enable','off');
    set(handles.text_skew,'enable','off');
    set(handles.text_kurt,'enable','off');
    set(handles.text_spent,'enable','off');
    set(handles.text_mode,'enable','off');
    set(handles.text_meanpitch,'enable','off');
    set(handles.text_maxpitch,'enable','off');
    set(handles.text_minpitch,'enable','off');
    
    set(handles.RecordTip,'enable','off');
    set(handles.textM1,'enable','off');
    set(handles.textM2,'enable','off');
    set(handles.textM3,'enable','off');
    set(handles.textM4,'enable','off');
    set(handles.textM5,'enable','off');
    set(handles.textM6,'enable','off');
    set(handles.textM7,'enable','off');
    set(handles.textM8,'enable','off');
    set(handles.textM9,'enable','off');
    set(handles.textM10,'enable','off');
    set(handles.textM11,'enable','off');
    set(handles.textM12,'enable','off');
    set(handles.textM13,'enable','off');
    set(handles.textF1,'enable','off');
    set(handles.textF2,'enable','off');
    set(handles.textF3,'enable','off');
    set(handles.textF4,'enable','off');
    set(handles.textF5,'enable','off');
    set(handles.textF6,'enable','off');
    set(handles.textF7,'enable','off');
    set(handles.textF8,'enable','off');
    set(handles.textF9,'enable','off');
    set(handles.textF10,'enable','off');
    set(handles.textF11,'enable','off');
    set(handles.textF12,'enable','off');
    set(handles.textF13,'enable','off');
    
    set(handles.textmeanmale1,'enable','off');
    set(handles.textmeanmale2,'enable','off');
    set(handles.textmeanmale3,'enable','off');
    set(handles.textmeanmale4,'enable','off');
    set(handles.textmeanmale5,'enable','off');
    set(handles.textmeanmale6,'enable','off');
    set(handles.textmeanmale7,'enable','off');
    set(handles.textmeanmale8,'enable','off');
    set(handles.textmeanmale9,'enable','off');
    set(handles.textmeanmale10,'enable','off');
    set(handles.textmeanmale11,'enable','off');
    set(handles.textmeanmale12,'enable','off');
    set(handles.textmeanmale13,'enable','off');
    set(handles.textmeanfemale1,'enable','off');
    set(handles.textmeanfemale2,'enable','off');
    set(handles.textmeanfemale3,'enable','off');
    set(handles.textmeanfemale4,'enable','off');
    set(handles.textmeanfemale5,'enable','off');
    set(handles.textmeanfemale6,'enable','off');
    set(handles.textmeanfemale7,'enable','off');
    set(handles.textmeanfemale8,'enable','off');
    set(handles.textmeanfemale9,'enable','off');
    set(handles.textmeanfemale10,'enable','off');
    set(handles.textmeanfemale11,'enable','off');
    set(handles.textmeanfemale12,'enable','off');
    set(handles.textmeanfemale13,'enable','off');
    
    set(handles.textcurrent1,'enable','off');
    set(handles.textcurrent2,'enable','off');
    set(handles.textcurrent3,'enable','off');
    set(handles.textcurrent4,'enable','off');
    set(handles.textcurrent5,'enable','off');
    set(handles.textcurrent6,'enable','off');
    set(handles.textcurrent7,'enable','off');
    set(handles.textcurrent8,'enable','off');
    set(handles.textcurrent9,'enable','off');
    set(handles.textcurrent10,'enable','off');
    set(handles.textcurrent11,'enable','off');
    set(handles.textcurrent12,'enable','off');
    set(handles.textcurrent13,'enable','off');
end
