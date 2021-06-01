classdef AlgorithmImplement < AlgorithmInterface
    %ALGORITHMIMPLEMENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %�̳���Ŀ�ӿ�
        %problemInterface;
    end
    
    properties
        %��������
        cacheData;
        %�����˲��в�
        dataZf;
        
        %�Դο�ʼ��
        trialStartPoint;
        
        %�����������ݳ���
        sampleCount;
        %ƫ�����ݳ���
        offsetLength;
        %����㷨
        method;
        
        %Ԥ�����˲���
        preprocessFilter;
        %ѡ�������
        selectChannel;
        %�Դ���ʼ�¼�����
        testTrialStartEvent;
        %ѵ��ģ��
        model
        %��ǰ��ԱID
        currentPersonId
    end
    
    
    methods
        function initial(obj)
            %��������ʣ���Ŀ�ļ��и���
            srate = 250;
            
            %ѡ������ţ����嵼�������Ŀ�ļ��и���
            obj.selectChannel = [26 29 30];
            
            %�Դ���ʼ�¼����壬��Ŀ˵���и���
            obj.testTrialStartEvent = 1;
            
            %����ʱ��
            calTime = 2;
            %
            obj.currentPersonId = 0;
            %����ƫ��ʱ�䣨s��
            offsetTime = 0;
            
            obj.offsetLength = floor(offsetTime * srate);
            
            obj.sampleCount = calTime * srate;
            
            obj.preprocessFilter = obj.getPreFilter(srate);
            
           %%obj.userDate = [];
            

            
            
        end
        
        function run(obj)
            
            %�Ƿ�ֹͣ��ǩ
            endFlag = false;
            
            %�Ƿ�������ģʽ��ǩ
            calFlag = false;
            
            while(~endFlag)
                
                dataModel = obj.problemInterface.getData();
                
                %�Ǽ���ģʽ������¼����
                if(~calFlag)
                    
                    calFlag = obj.idleProcess(dataModel);
                    
                else
                    %����ģʽ������д���
                    [calFlag,resultType] = obj.calculateProcess(dataModel);
                    
                    if(~isempty(resultType))
                        %����н��������б���
                        reportModel = ReportModel();
                        reportModel.resultType = resultType;
                        
                        obj.problemInterface.report(reportModel);
                        
                        %ͬʱ��ջ���
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
            %���ݱ��Լ�����Ӧ��ѵ��ģ��
            personID = dataModel.personID;
            if(obj.currentPersonId~=personID)
                obj.currentPersonId = obj.currentPersonId + 1;   
                %��ʼ���㷨
                import py.mi_classifier.*;
				obj.method = miClassifier(obj.currentPersonId); 
        
            end
            eventData = data(end,:);
            eventPosition = find(eventData == obj.testTrialStartEvent,1);
            eegData = data(1:end-1,:);
            
            %���eventΪ�գ���ʾ��Ȼ�ڵ�ǰ�Դ��У��������ݳ����ж��Ƿ����
            if(isempty(eventPosition))
                cacheDataLength = size(obj.cacheData,2);
                
                %����������ݳ��ȴﵽҪ������м���
                if(size(eegData,2)> obj.sampleCount - cacheDataLength)
                    obj.cacheData = cat(2,obj.cacheData,eegData(:,1:obj.sampleCount - cacheDataLength));
                    usedData = double(obj.cacheData(:,obj.offsetLength+1:end));
                    %�˲�����
                    usedData = obj.preprocess(usedData);
                    %tags=?
                    %��ʼ����
                    %train(usedData,tags)
                    result = obj.method.recognize(usedData);
                    resultType = result.double;
                    
                    %ֹͣ����ģʽ
                    calFlag = false;
                else
                    %��֮�����ɼ�����
                    obj.cacheData = cat(2,obj.cacheData,eegData);
                    resultType = [];
                    calFlag = true;
                end
                
            else
                %event�ǿգ���ʾ��һ�Դ��ѿ�ʼ����Ҫǿ�ƽ�������
                nextTrialStartPoint = eventPosition(1);
                cacheDataLength = size(obj.cacheData,2);
                usedLength = min([nextTrialStartPoint,obj.sampleCount - cacheDataLength]);
                
                obj.cacheData = cat(2,obj.cacheData,data(1:end-1,1:usedLength));
                usedData = double(obj.cacheData(:,obj.offsetLength+1:end));
                %�˲�����
                usedData = obj.preprocess(usedData);
                %��ʼ����
                resultType = obj.method.recognize(usedData);
                %��ʼ���Դεļ���ģʽ
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

