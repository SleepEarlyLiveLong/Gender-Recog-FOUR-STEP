% Name:     GenderRecogGUI 0.2
% Function: Recognition of speaker's gender based on his/her speech signal.
%           The COMPLETE version of GenderRecogGUI 0.1

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

% Last Modified by GUIDE v2.5 02-Apr-2019 10:32:43

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
control.classifier = 'NaiveBayes';        % 分类器名称
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
control.featurechanged = zeros(1,8);       % 特征是否已经改变 0-没有改变 1-已经改变
control.CurrentSplitRatioChoice = 3;               % 默认选7/3的分割比例
control.CurrentClassifierChoice = 1;               % 默认分类器 DistCompare
control.PastSplitRatioChoice = 3;
control.PastClassifierChoice = 1;
control.choiceline = [0 0 0];
control.wronglist = [];
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

set(handles.checkbox_meanpitch,'Value',1);
set(handles.checkbox_maxpitch,'Value',1);
set(handles.checkbox_minpitch,'Value',1);
set(handles.checkbox_mfcc,'Value',0);

% 默认比例是7:3
set(handles.MenuSeventyThirty,'Checked','on');
% 默认分类器是 NaiveBayes 朴素贝叶斯
set(handles.MenuNaiveBayes,'Checked','on');

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
    file_list = cell(2,1);
    for i=1:category_num
        VOICES(i).name = control.fold_list(i+2).name;
        % 每个类中的若干个文件
        file_list{i} = dir([control.datasetpath,'\',VOICES(i).name]);
        VOICES(i).num = min(length(file_list{i})-2,20+round(5*rand()));   % 随机抽取部分数据
%         VOICES(i).num = length(file_list{i})-2;                            % 所有数据
        for j=1:VOICES(i).num
            [currentvoice,fs] = audioread([control.datasetpath,'\',VOICES(i).name,'\',file_list{i}(j+2).name]);
            allfts = myfeature(currentvoice,fs);        % 提取各种特征
            voiced = mygetvoiced(currentvoice,fs);      % 把浊音提取出来
            % 根据用户的选择提取特征
            feature = [];
            if get(handles.checkbox_meanpitch,'Value')==1       % 选择了 meanpitch
                feature = [feature allfts(11)];
                control.choiceline(1) = 1;
            end
            if get(handles.checkbox_maxpitch,'Value')==1       % 选择了 maxpitch 
                feature = [feature allfts(12)];
                control.choiceline(2) = 1;
            end
            if get(handles.checkbox_minpitch,'Value')==1       % 选择了 minpitch 
                feature = [feature allfts(13)];
                control.choiceline(3) = 1;
            end
            if get(handles.checkbox_mfcc,'Value')==1           % 选择了 mfcc
                feature = [feature allfts(14:26)];
            end
            if get(handles.checkbox_feature1,'Value')==1       % 选择了 feature1 
                feature = [feature yourfeature1(voiced,fs)];
            end
            if get(handles.checkbox_feature2,'Value')==1       % 选择了 feature2
                feature = [feature yourfeature2(voiced,fs)];
            end
            if get(handles.checkbox_feature3,'Value')==1       % 选择了 feature3
                feature = [feature yourfeature3(voiced,fs)];
            end
            if get(handles.checkbox_feature4,'Value')==1       % 选择了 feature4
                feature = [feature yourfeature4(voiced,fs)];
            end
            if isempty(feature)
                allWidgetEnable(handles,1);
                errordlg('错误! 你至少需要选择一种特征','错误提示');
                return;
            end
            VOICES(i).data(j,:) = feature;          % 转为行向量存入结构体中
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
    train_index_remember = [];
    for i=1:category_num
        file_num = VOICES(i).num;
        [val_index,train_index] = crossvalind('holdOut',file_num,control.TrainSegRatio);
        traindata = [traindata;VOICES(i).data(train_index,:)];
        trainlabel = [trainlabel;i*ones(sum(train_index),1)];
        valdata = [valdata;VOICES(i).data(val_index,:)];
        vallabel = [vallabel;i*ones(sum(val_index),1)];
        train_index_remember = [train_index_remember;train_index];
    end
    % 一些参数
    male_train_num = sum(train_index_remember(1:size(VOICES(1).data,1)));
    male_val_num = size(VOICES(1).data,1) - male_train_num;
    female_train_num = sum( train_index_remember(size(VOICES(1).data,1)+1:end) );
    female_val_num = size(VOICES(2).data,1) - female_train_num;
    
    control.traindata = traindata;
    control.trainlabel = trainlabel;
    control.valdata = valdata;
    control.vallabel = vallabel;
    fprintf('训练集/验证集划分完毕\n');
    %% 训练
    % 开始训练分类器
    if strcmp(control.classifier,'DistCompare')==1
        % 训练测试
        Results = myDistDetermine(control.traindata,control.trainlabel,control.valdata);
        cfsmtx = mycfsmtx(control.vallabel,Results.predicted_label);
        % 将识别错误的数据的序号找到
        control.wronglist = find((control.vallabel-Results.predicted_label)~=0);
        % 数据保存
        classifierparam.DCtraincore = Results.DCtraincore;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'NaiveBayes')==1
        % 量化预处理
        ftsnum = size(control.traindata,2);     % 特征数值
        traindata_qualify = zeros(size(control.traindata));
        valdata_qualify = zeros(size(control.valdata));
        for i=1:ftsnum
            traindata_qualify(:,i) = mydiscretization(control.traindata(:,i),control.ftsRange);     % 特征数值量化阶ftsRange = 20;
            valdata_qualify(:,i) = mydiscretization(control.valdata(:,i),control.ftsRange);
        end
        % 训练测试
        [TrainingSets,ValidationSets] = myNaiveBayesTrain(traindata_qualify,control.trainlabel,...
            valdata_qualify,control.vallabel,control.ftsRange);
        predicted_label = myNaiveBayesValidation(TrainingSets,ValidationSets);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 将识别错误的数据的序号找到
        control.wronglist = find((control.vallabel-predicted_label)~=0);
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
        [~,traindata_new_f,~,~,~] = mykmeans( control.traindata(1:train_female_num,:),control.clusters );       % clusters = 100;
        [~,traindata_new_m,~,~,~] = mykmeans( control.traindata(train_female_num+1:train_female_num+train_male_num,:),control.clusters );
        traindata_new = [traindata_new_f;traindata_new_m];
        trainlabel_new = [ones(control.clusters,1);2*ones(control.clusters,1)];
        % 训练测试
        KMSKNN_model = fitcknn(traindata_new,trainlabel_new,'NumNeighbors',control.Knearest); % Knearest=12
        predicted_label = KMSKNN_model.predict(control.valdata);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 将识别错误的数据的序号找到
        control.wronglist = find((control.vallabel-predicted_label)~=0);
        % 数据保存
        classifierparam.KMSKNNmodel = KMSKNN_model;
        % 面板显示
        control.ValCorrRate = cfsmtx(end,end);             % 准确率
        control.RecallRate = cfsmtx(1,end);             % 召回率
        set(handles.textCurrentAccuracy,'string',num2str(control.ValCorrRate,'%.3f'));
        set(handles.textCurrentRecallRate,'string',num2str(control.RecallRate,'%.3f'));
    elseif strcmp(control.classifier,'KNN')==1
        % 训练测试
        KNN_model = fitcknn(control.traindata,control.trainlabel,'NumNeighbors',7); 
        predicted_label = KNN_model.predict(control.valdata);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 将识别错误的数据的序号找到
        control.wronglist = find((control.vallabel-predicted_label)~=0);
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
        [~,c_temp,g_temp] = SVMcgForClass(control.trainlabel,control.traindata,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
        [cmin,cmax,gmin,gmax,v,cstep,gstep,accstep] = deal(log(c_temp)/log(2)-2,log(c_temp)/log(2)+2,...
            log(g_temp)/log(2)-2,log(g_temp)/log(2)+2,3,0.25,0.25,0.5);
        [~,bestc,bestg] = SVMcgForClass(control.trainlabel,control.traindata,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep);
        % 训练测试
        cmd = [' -s ',num2str(0),' -c ',num2str(bestc),' -g ',num2str(bestg)];
        svm_model = svmtrain(control.trainlabel,control.traindata,cmd);
        [predicted_label, ~, ~] = svmpredict(control.vallabel,control.valdata,svm_model);
        cfsmtx = mycfsmtx(control.vallabel,predicted_label);
        % 将识别错误的数据的序号找到
        control.wronglist = find((control.vallabel-predicted_label)~=0);
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
    
    %% 训练结束后的第一件事：找到分类错误的数据序号
    % 找到分类错误的语音序号（只看验证集）
%     control.wronglist
    findval_index = find(train_index_remember==0);
    findvalwrong_index = findval_index(control.wronglist);
    findfemalevalwrong_index = findvalwrong_index(findvalwrong_index<=VOICES(1).num);
    findmalevalwrong_index = findvalwrong_index(findvalwrong_index>VOICES(1).num);
    findmalevalwrong_index = findmalevalwrong_index-VOICES(1).num;
    % 先删除文件夹中原本存在的文件
    delete([pwd,'\GUIClassWrng\female\*.*']);
    delete([pwd,'\GUIClassWrng\male\*.*']);
    % 将分类错误的语音拷贝到文件夹中
    for i=1:length(findfemalevalwrong_index)
        source = [control.datasetpath,'\female\',file_list{1,1}(findfemalevalwrong_index(i)+2).name];
        destination = 'GUIClassWrng\female\';
        copyfile(source,destination);
    end
    for i=1:length(findmalevalwrong_index)
        source = [control.datasetpath,'\male\',file_list{2,1}(findmalevalwrong_index(i)+2).name];
        destination = 'GUIClassWrng\male\';
        copyfile(source,destination);
    end
    %% 训练阶段，复选框后面的静态文本框（中间）一律不显示特征数据
    % 11 text_meanpitch
    if control.choiceline(1)==1
        set(handles.text_meanpitch,'string','meanpitch');
    else
        set(handles.text_meanpitch,'string','——');
    end
    % 12 text_maxpitch
    if control.choiceline(2)==1
        set(handles.text_maxpitch,'string','maxpitch');
    else
        set(handles.text_maxpitch,'string','——');
    end
    % 13 text_minpitch
    if control.choiceline(3)==1
        set(handles.text_minpitch,'string','minpitch');  
    else
        set(handles.text_minpitch,'string','——');
    end
    
    %% 训练阶段，复选框后面的静态文本框（上下）显示男女声在此项上的均值
    TraindataSelected4male = control.traindata;
    TraindataSelected4male(control.trainlabel==1,:) = [];
    TraindataSelected4male = mean(TraindataSelected4male);
    TraindataSelected4female = control.traindata;
    TraindataSelected4female(control.trainlabel==2,:) = [];
    TraindataSelected4female = mean(TraindataSelected4female);
    
    temp_TraindataSelected4male = TraindataSelected4male;
    temp_TraindataSelected4female = TraindataSelected4female;
    if control.choiceline(1) == 1
        set(handles.textM1,'string',num2str(temp_TraindataSelected4male(1)));
        set(handles.textF1,'string',num2str(temp_TraindataSelected4female(1)));
        temp_TraindataSelected4male(1) = [];
        temp_TraindataSelected4female(1) = [];
    else
        set(handles.textM1,'string','——');
        set(handles.textF1,'string','——');
    end
    if control.choiceline(2) == 1
        set(handles.textM2,'string',num2str(temp_TraindataSelected4male(1)));
        set(handles.textF2,'string',num2str(temp_TraindataSelected4female(1)));
        temp_TraindataSelected4male(1) = [];
    else
        set(handles.textM2,'string','——');
        set(handles.textF2,'string','——');
        set(handles.text_maxpitch,'string','——');
    end
    if control.choiceline(3) == 1
        set(handles.textM3,'string',num2str(temp_TraindataSelected4male(1)));
        set(handles.textF3,'string',num2str(temp_TraindataSelected4female(1)));
    else
        set(handles.textM3,'string','——');
        set(handles.textF3,'string','——');
    end
    clear temp_TraindataSelected4male;
    clear temp_TraindataSelected4female;

    %% 标志位
    % 训练结束后，所有特征改变标志位清零
    control.featurechanged = zeros(1,8);
    % 训练结束后，选择的划分比例由 current 状态变为 past（当前变成上一轮）
    control.PastSplitRatioChoice = control.CurrentSplitRatioChoice;
    control.PastClassifierChoice = control.CurrentClassifierChoice;
end

% % 1-2 读入语音--------------------------------------------------------------------
% function MenuGetVoice_Callback(hObject, eventdata, handles)
% % hObject    handle to MenuGetVoice (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)


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
hs = msgbox({'软件版本说明:';'';'Version: 0.2 ';'';...
             'Author: Chen Tianyang';'';...
             'Data:2019-03-20';''},'Version Information');
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
%                        3、 评价指标                       
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- 查看分类错误的结果 打开文件夹读取数据
function pushbutton_check_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_check (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
[filename,pathname] = uigetfile({'*.wav';'*.mp3';'*.ogg';'*.au';'*.flac'},...
    'Select a voice file',[pwd,'\GUIClassWrng\']);
if ischar(filename) && ischar(pathname)
    [x,fs] = audioread([pathname,filename]);
    x = resample(x,8000,fs);
    fs = 8000;
    x = x/max(abs(x));
    recorder.voice = x;
    param.fs = fs;
    % 绘图
    axes(handles.axes_time);
    t = (0:length(x)-1)/fs;
    plot(t,x);xlabel('Time/s');ylabel('Amplitude');title('Speech waveform');
    % 显示栏的结论和数据清除
    set(handles.textCurrentGender,'String','性别');
    % 11 text_meanpitch
    set(handles.text_meanpitch,'string','meanpitch');
    % 12 text_maxpitch
    set(handles.text_maxpitch,'string','maxpitch');
    % 13 text_minpitch
    set(handles.text_minpitch,'string','minpitch');  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         4、语音性别识别                      
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
% 11 text_meanpitch
set(handles.text_meanpitch,'string','meanpitch');
% 12 text_maxpitch
set(handles.text_maxpitch,'string','maxpitch');
% 13 text_minpitch
set(handles.text_minpitch,'string','minpitch');  

% 2 菜单栏-读入语音 / 从本地加载语音信号按钮 --------------------------------------------
function ChooseRecord_Callback(hObject, eventdata, handles)
% hObject    handle to ChooseRecord (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global recorder;
global param;
[filename,pathname] = uigetfile({'*.wav';'*.mp3';'*.ogg';'*.au';'*.flac'},'Select a voice file');
if ischar(filename)&&ischar(pathname)
    [x,fs] = audioread([pathname,filename]);
    if fs ~= 8000
        x = resample(x,8000,fs);
        fs = 8000;
    end
    x = x/max(abs(x));
    recorder.voice = x;
    param.fs = fs;
    % 绘图
    axes(handles.axes_time);
    t = (0:length(x)-1)/fs;
    plot(t,x);xlabel('Time/s');ylabel('Amplitude');title('Speech waveform');
    % 显示栏等4个控件上的结论和数据清除(11 text_meanpitch，12 text_maxpitch，13 text_minpitch)
    set(handles.textCurrentGender,'String','性别');
    set(handles.text_meanpitch,'string','meanpitch');
    set(handles.text_maxpitch,'string','maxpitch');
    set(handles.text_minpitch,'string','minpitch');  
    % 提示信息
    hs = msgbox('语音数据已读取完毕','提示');
    ht = findobj(hs, 'Type', 'text');     
    set(ht,'FontSize',8);     
    set(hs, 'Resize', 'on'); 
else
    warndlg('警告！您没有选择测试语音文件，请选择任一语音文件','警告提示');
end

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
elseif sum(control.featurechanged)~=0
    errordlg('错误！特征组合已改变，必须重新训练','错误提示');
elseif control.CurrentSplitRatioChoice ~= control.PastSplitRatioChoice
    errordlg('错误！训练集验证集比例已改变，必须重新训练','错误提示');
elseif control.CurrentClassifierChoice ~= control.PastClassifierChoice
    errordlg('错误！分类器已改变，必须重新训练','错误提示');
else
    allfts = myfeature(recorder.voice,param.fs);        % 提取各种特征
    voiced = mygetvoiced(recorder.voice,param.fs);      % 把浊音提取出来
    % 根据用户的选择提取特征
    feature = [];
    if get(handles.checkbox_meanpitch,'Value')==1       % 选择了 meanpitch 
        feature = [feature allfts(11)];
    end
    if get(handles.checkbox_maxpitch,'Value')==1       % 选择了 maxpitch 
        feature = [feature allfts(12)];
    end
    if get(handles.checkbox_minpitch,'Value')==1       % 选择了 minpitch 
        feature = [feature allfts(13)];
    end
    if get(handles.checkbox_mfcc,'Value')==1           % 选择了 minpitch 
        feature = [feature allfts(14:26)];
    end
    if get(handles.checkbox_feature1,'Value')==1       % 选择了 feature1 
        feature = [feature myfeature1(voiced,fs)];
    end
    if get(handles.checkbox_feature2,'Value')==1       % 选择了 feature2
        feature = [feature myfeature2(voiced,fs)];
    end
    if get(handles.checkbox_feature3,'Value')==1       % 选择了 feature3
        feature = [feature myfeature3(voiced,fs)];
    end
    if get(handles.checkbox_feature4,'Value')==1       % 选择了 feature4
        feature = [feature myfeature4(voiced,fs)];
    end    
    % 判断测试的特征和训练/验证的特征是否一样
    % 判断当前采用的是哪种分类器
    if strcmp(control.classifier,'DistCompare')==1
        if isempty(classifierparam.DCtraincore)
            errordlg('错误！Dist Comapre 分类器不存在,请先训练此分类器','错误提示');
        else
            % DistCompare 检测
            predicted_label = myDistDetermineTest(classifierparam.DCtraincore,feature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','女');
            else
                set(handles.textCurrentGender,'string','男');
            end
        end
    elseif strcmp(control.classifier,'NaiveBayes')==1
        if isempty(classifierparam.NBTrainingSets)||isempty(classifierparam.NBValidationSets)
            errordlg('错误！Naive Bayes 分类器不存在,请先训练此分类器','错误提示');
        else 
            % 首先从 NBTrainingSets 中得出每个特征的最大最小值
            feature_max = max([control.traindata;control.valdata]);
            feature_min = min([control.traindata;control.valdata]);
            % 以此为依据对新特征进行增量量化
            feature_qualify = mydiscretization2(feature_max,feature_min,control.ftsRange,feature);
            % NaiveBayes 检测
            [predicted_label,female_prob,male_prob] = myNaiveBayesTest(classifierparam.NBTrainingSets,feature_qualify);
            if predicted_label==1
                set(handles.textCurrentGender,'string','女');
            else
                set(handles.textCurrentGender,'string','男');
            end
            set(handles.text_PF,'string',num2str(female_prob));
            set(handles.text_PM,'string',num2str(male_prob));
        end
    elseif strcmp(control.classifier,'KMSKNN')==1
        if isempty(classifierparam.KMSKNNmodel)
            errordlg('错误！KMSKNN 分类器不存在,请先训练此分类器','错误提示');
        else
            KMSKNN_model = classifierparam.KMSKNNmodel;
            predicted_label = KMSKNN_model.predict(feature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','女');
            else
                set(handles.textCurrentGender,'string','男');
            end
        end
    elseif strcmp(control.classifier,'KNN')==1
        if isempty(classifierparam.KNNmodel)
            errordlg('错误！KNN 分类器不存在,请先训练此分类器','错误提示');
        else
            KNN_model = classifierparam.KNNmodel;
            predicted_label = KNN_model.predict(feature);
            if predicted_label==1
                set(handles.textCurrentGender,'string','女');
            else
                set(handles.textCurrentGender,'string','男');
            end
        end
    elseif strcmp(control.classifier,'SVM')==1
        if isempty(classifierparam.SVMmodel)
            errordlg('错误！SVM 分类器不存在,请先训练此分类器','错误提示');
        else
            % 测试
            [predicted_label, ~, ~] = svmpredict(1,feature,classifierparam.SVMmodel);   % 第一个参数"1"是待判断数据的伪标签
            if predicted_label==1
                set(handles.textCurrentGender,'string','女');
            else
                set(handles.textCurrentGender,'string','男');
            end
        end
    else
        errordlg('错误！该分类器不存在','错误提示');
    end
    currentfeature = feature;
    if control.choiceline(1)==1
        set(handles.text_meanpitch,'string',num2str(currentfeature(1)));
        currentfeature(1) = [];
    else
        set(handles.text_meanpitch,'string','——');
    end
    if control.choiceline(2)==1
        set(handles.text_maxpitch,'string',num2str(currentfeature(1)));
        currentfeature(1) = [];
    else
        set(handles.text_maxpitch,'string','——');
    end
    if control.choiceline(3)==1
        set(handles.text_minpitch,'string',num2str(currentfeature(1)));
    else
        set(handles.text_minpitch,'string','——');
    end
    clear currentfeature;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                         5、坐标面板
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
%                         6、特征选择
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- 1 meanpitch.
function checkbox_meanpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_meanpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_meanpitch
global control;
if control.featurechanged(1)==0
    control.featurechanged(1) = 1;
else
    control.featurechanged(1) = 0;
end
if get(hObject,'value')==1
    control.choiceline(1) = 1;
else
    control.choiceline(1) = 0;
end

% --- 2 maxpitch.
function checkbox_maxpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_maxpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_maxpitch
global control;
if control.featurechanged(2)==0
    control.featurechanged(2) = 1;
else
    control.featurechanged(2) = 0;
end
if get(hObject,'value')==1
    control.choiceline(2) = 1;
else
    control.choiceline(2) = 0;
end

% --- 3 minpitch.
function checkbox_minpitch_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_minpitch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_minpitch
global control;
if control.featurechanged(3)==0
    control.featurechanged(3) = 1;
else
    control.featurechanged(3) = 0;
end
if get(hObject,'value')==1
    control.choiceline(3) = 1;
else
    control.choiceline(3) = 0;
end

% --- 4 mfcc.
function checkbox_mfcc_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_mfcc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_mfcc
global control;
if control.featurechanged(4)==0
    control.featurechanged(4) = 1;
else
    control.featurechanged(4) = 0;
end

% --- 5 feature1.
function checkbox_feature1_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_feature1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_feature1
global control;
if control.featurechanged(5)==0
    control.featurechanged(5) = 1;
else
    control.featurechanged(5) = 0;
end

% --- 6 feature2.
function checkbox_feature2_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_feature2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_feature2
global control;
if control.featurechanged(6)==0
    control.featurechanged(6) = 1;
else
    control.featurechanged(6) = 0;
end

% --- 7 feature3.
function checkbox_feature3_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_feature3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_feature3
global control;
if control.featurechanged(7)==0
    control.featurechanged(7) = 1;
else
    control.featurechanged(7) = 0;
end

% --- 8 feature4.
function checkbox_feature4_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_feature4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_feature4
global control;
if control.featurechanged(8)==0
    control.featurechanged(8) = 1;
else
    control.featurechanged(8) = 0;
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
    set(handles.pushbutton_check,'enable','on');
    set(handles.StartRecord,'enable','on');
    set(handles.ChooseRecord,'enable','on');
    set(handles.pushbuttonGoJudge,'enable','on');
    set(handles.pushbuttonPlay,'enable','on');
    set(handles.textCurrentGender,'enable','on');

    set(handles.text_PM,'enable','on');
    set(handles.text_PF,'enable','on');
    set(handles.checkbox_meanpitch,'enable','on');
    set(handles.checkbox_maxpitch,'enable','on');
    set(handles.checkbox_minpitch,'enable','on');
    set(handles.checkbox_mfcc,'enable','on');
    set(handles.checkbox_feature1,'enable','on');
    set(handles.checkbox_feature2,'enable','on');
    set(handles.checkbox_feature3,'enable','on');
    set(handles.checkbox_feature4,'enable','on');

    set(handles.text_meanpitch,'enable','on');
    set(handles.text_maxpitch,'enable','on');
    set(handles.text_minpitch,'enable','on');
    
    set(handles.RecordTip,'enable','on');

    set(handles.textM1,'enable','on');
    set(handles.textM2,'enable','on');
    set(handles.textM3,'enable','on');

    set(handles.textF1,'enable','on');
    set(handles.textF2,'enable','on');
    set(handles.textF3,'enable','on');
    
    set(handles.textmeanmale1,'enable','on');
    set(handles.textmeanmale2,'enable','on');
    set(handles.textmeanmale3,'enable','on');

    set(handles.textmeanfemale1,'enable','on');
    set(handles.textmeanfemale2,'enable','on');
    set(handles.textmeanfemale3,'enable','on');
    
    set(handles.textcurrent1,'enable','on');
    set(handles.textcurrent2,'enable','on');
    set(handles.textcurrent3,'enable','on');

else
    set(handles.textCurrentAccuracy,'enable','off');
    set(handles.textCurrentRecallRate,'enable','off');
    set(handles.pushbutton_check,'enable','off');
    set(handles.StartRecord,'enable','off');
    set(handles.ChooseRecord,'enable','off');
    set(handles.pushbuttonGoJudge,'enable','off');
    set(handles.pushbuttonPlay,'enable','off');
    set(handles.textCurrentGender,'enable','off');

    set(handles.text_PM,'enable','off');
    set(handles.text_PF,'enable','off');
    set(handles.checkbox_meanpitch,'enable','off');
    set(handles.checkbox_maxpitch,'enable','off');
    set(handles.checkbox_minpitch,'enable','off');
    set(handles.checkbox_mfcc,'enable','off');
    set(handles.checkbox_feature1,'enable','off');
    set(handles.checkbox_feature2,'enable','off');
    set(handles.checkbox_feature3,'enable','off');
    set(handles.checkbox_feature4,'enable','off');
    
    set(handles.text_meanpitch,'enable','off');
    set(handles.text_maxpitch,'enable','off');
    set(handles.text_minpitch,'enable','off');
    
    set(handles.RecordTip,'enable','off');

    set(handles.textM1,'enable','off');
    set(handles.textM2,'enable','off');
    set(handles.textM3,'enable','off');

    set(handles.textF1,'enable','off');
    set(handles.textF2,'enable','off');
    set(handles.textF3,'enable','off');
    
    set(handles.textmeanmale1,'enable','off');
    set(handles.textmeanmale2,'enable','off');
    set(handles.textmeanmale3,'enable','off');

    set(handles.textmeanfemale1,'enable','off');
    set(handles.textmeanfemale2,'enable','off');
    set(handles.textmeanfemale3,'enable','off');
    
    set(handles.textcurrent1,'enable','off');
    set(handles.textcurrent2,'enable','off');
    set(handles.textcurrent3,'enable','off');
end
