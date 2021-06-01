classdef DataModel < handle
    %DATAMODULE 传递接收数据模型
    %   Detailed explanation goes here
    
    properties
        
        %double矩阵类型,为(L+1)*N维矩阵，其中L表示通道数,最后一个通道为Trigger信号，N表示数据包内样本
        data;
        
        %double类型:表示当前数据块相对本次Session起点位置(每名受试者一次采集视为一个Session)
        %重置为0,则认为开始新的block
        startPosition;
        
        %double类型
        personID;

        %程序终止标志
        %bool类型
        finishedFlag;
        
    end
    
    methods
    end
    
end

 