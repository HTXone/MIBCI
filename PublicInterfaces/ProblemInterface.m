classdef ProblemInterface < handle
    methods(Abstract)      
        %��ȡ���ݽӿڣ�����DataModel���ͱ���
        dataModel = getData(obj);        
        %�������ӿڣ�����ReportModel���ͱ���
        report(obj,reportModel);       
    end
end

