classdef AlgorithmImplement < AlgorithmInterface
    %ALGORITHMIMPLEMENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %继承题目接口
        %problemInterface;
    end
    
    properties
        %缓存数据
        cacheData;
        %迭代滤波残差
        dataZf;
        
        %试次开始点
        trialStartPoint;
        
        %计算所用数据长度
        sampleCount;
        %偏移数据长度
        offsetLength;
        %求解算法
        method;
        
        %预处理滤波器
        preprocessFilter;
        %选择导联序号
        selectChannel;
        %试次启始事件定义
        testTrialStartEvent;
        %训练模型
        model
        %当前人员ID
        currentPersonId
    end
    
    
    methods
        function initial(obj)
            %定义采样率，题目文件中给出
            srate = 250;
            
            %选择导联编号，具体导联编号题目文件中给出
            obj.selectChannel = [26 29 30];
            
            %试次启始事件定义，题目说明中给出
            obj.testTrialStartEvent = 1;
            
            %计算时间
            calTime = 2;
            %
            obj.currentPersonId = 0;
            %计算偏移时间（s）
            offsetTime = 0;
            
            obj.offsetLength = floor(offsetTime * srate);
            
            obj.sampleCount = calTime * srate;
            
            obj.preprocessFilter = obj.getPreFilter(srate);
            
           %%obj.userDate = [];
            

            
            
        end
        
        function run(obj)
            
            %是否停止标签
            endFlag = false;
            
            %是否进入计算模式标签
            calFlag = false;
            
            while(~endFlag)
                
                dataModel = obj.problemInterface.getData();
                
                %非计算模式则进行事件检测
                if(~calFlag)
                    
                    calFlag = obj.idleProcess(dataModel);
                    
                else
                    %计算模式，则进行处理
                    [calFlag,resultType] = obj.calculateProcess(dataModel);
                    
                    if(~isempty(resultType))
                        %如果有结果，则进行报告
                        reportModel = ReportModel();
                        reportModel.resultType = resultType;
                        
                        obj.problemInterface.report(reportModel);
                        
                        %同时清空缓存
                        obj.clearCach();
                    end
                    
                end
                
                endFlag = dataModel.finishedFlag;
                
            end
            
        end
        
    end
    methods(Access = private)
        function [calFlag] = idleProcess(obj,dataModel)
            data = dataModel.data;
            eventData = data(end,:);
            eventPosition = find(eventData == obj.testTrialStartEvent,1);
            eegData = data(1:end-1,:);
            if(~isempty(eventPosition))
                calFlag = true;
                obj.trialStartPoint = eventPosition(1);
                
                obj.cacheData = eegData(:,obj.trialStartPoint:end);
                
            else
                calFlag = false;
                obj.trialStartPoint = [];
                obj.clearCach();
            end
            
        end
        
        function [calFlag,resultType] = calculateProcess(obj,dataModel)
            
            data = dataModel.data;
            %根据被试加载相应的训练模型
            personID = dataModel.personID;
            if(obj.currentPersonId~=personID)
                obj.currentPersonId = obj.currentPersonId + 1;   
                %初始化算法
                import py.mi_classifier.*;
				obj.method = miClassifier(obj.currentPersonId); 
        
            end
            eventData = data(end,:);
            eventPosition = find(eventData == obj.testTrialStartEvent,1);
            eegData = data(1:end-1,:);
            
            %如果event为空，表示依然在当前试次中，根据数据长度判断是否计算
            if(isempty(eventPosition))
                cacheDataLength = size(obj.cacheData,2);
                
                %如果接收数据长度达到要求，则进行计算
                if(size(eegData,2)> obj.sampleCount - cacheDataLength)
                    obj.cacheData = cat(2,obj.cacheData,eegData(:,1:obj.sampleCount - cacheDataLength));
                    usedData = double(obj.cacheData(:,obj.offsetLength+1:end));
                    %滤波处理
                    usedData = obj.preprocess(usedData);
                    %tags=?
                    %开始计算
                    %train(usedData,tags)
                    result = obj.method.recognize(usedData);
                    resultType = result.double;
                    
                    %停止计算模式
                    calFlag = false;
                else
                    %反之继续采集数据
                    obj.cacheData = cat(2,obj.cacheData,eegData);
                    resultType = [];
                    calFlag = true;
                end
                
            else
                %event非空，表示下一试次已开始，需要强制结束计算
                nextTrialStartPoint = eventPosition(1);
                cacheDataLength = size(obj.cacheData,2);
                usedLength = min([nextTrialStartPoint,obj.sampleCount - cacheDataLength]);
                
                obj.cacheData = cat(2,obj.cacheData,data(1:end-1,1:usedLength));
                usedData = double(obj.cacheData(:,obj.offsetLength+1:end));
                %滤波处理
                usedData = obj.preprocess(usedData);
                %开始计算
                resultType = obj.method.recognize(usedData);
                %开始新试次的计算模式
                calFlag = true;
            end
            
        end
        
    end
    
    
    methods(Access = private)
        function preprocessFilter = getPreFilter(~,srate)
            Fo = 50;
            Q = 35;
            BW = (Fo/(srate/2))/Q;
            [preprocessFilter.B,preprocessFilter.A] = iircomb(srate/Fo,BW,'notch');
        end
        function clearCach(obj)
            obj.cacheData = [];
            obj.dataZf = [];
        end
        function data = preprocess(obj,data)
            
            data = data(obj.selectChannel,:);
            
            data = filtfilt(obj.preprocessFilter.B,obj.preprocessFilter.A,data.');
            
            data = data.';
            
        end
    end
    
end

