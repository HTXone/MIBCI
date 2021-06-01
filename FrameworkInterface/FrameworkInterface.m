classdef FrameworkInterface < handle
    %FRAMEWORKINTERFACE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Abstract)
        %��ʼ������
        initial(obj);
        
        %������ݣ�personDataTransferModelSetΪPersonDataTransferModel��������
        addData(obj,personDataTransferModelSet);
        
        %�����������
        clearData(obj);
        
        %����㷨
        addAlgorithm(obj);
        
        %�����㷨
        run(obj);
        
        %��ȡ�ɼ�������ֵΪScoreModel���Ͷ���
        scoreModel = getScore(obj);
        
        %�����ǰ�㷨���н����Ϊ��һ���㷨��׼��
        clearAlgorithm(obj);
        
    end
    
end

