classdef AlgorithmInterface < handle
    properties
        problemInterface;
    end
    
    methods(Abstract)
        %�㷨��ʼ������
        initial(obj);
        %�㷨ִ������
        run(obj);     
    end
    methods
        function setProblemInterface(obj,problemInterface)  
            obj.problemInterface = problemInterface;
        end
    end
    
end

