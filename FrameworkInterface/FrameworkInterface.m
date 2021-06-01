classdef FrameworkInterface < handle
    %FRAMEWORKINTERFACE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Abstract)
        %初始化函数
        initial(obj);
        
        %填充数据，personDataTransferModelSet为PersonDataTransferModel类型数组
        addData(obj,personDataTransferModelSet);
        
        %清空已有数据
        clearData(obj);
        
        %填充算法
        addAlgorithm(obj);
        
        %运行算法
        run(obj);
        
        %获取成绩，返回值为ScoreModel类型对象
        scoreModel = getScore(obj);
        
        %清除当前算法所有结果，为下一个算法做准备
        clearAlgorithm(obj);
        
    end
    
end

