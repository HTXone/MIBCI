classdef DataModel < handle
    %DATAMODULE ���ݽ�������ģ��
    %   Detailed explanation goes here
    
    properties
        
        %double��������,Ϊ(L+1)*Nά��������L��ʾͨ����,���һ��ͨ��ΪTrigger�źţ�N��ʾ���ݰ�������
        data;
        
        %double����:��ʾ��ǰ���ݿ���Ա���Session���λ��(ÿ��������һ�βɼ���Ϊһ��Session)
        %����Ϊ0,����Ϊ��ʼ�µ�block
        startPosition;
        
        %double����
        personID;

        %������ֹ��־
        %bool����
        finishedFlag;
        
    end
    
    methods
    end
    
end

 