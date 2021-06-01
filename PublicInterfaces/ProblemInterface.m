classdef ProblemInterface < handle
    methods(Abstract)      
        %获取数据接口，返回DataModel类型变量
        dataModel = getData(obj);        
        %结果报告接口，输入ReportModel类型变量
        report(obj,reportModel);       
    end
end

