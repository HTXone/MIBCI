classdef AlgorithmInterface < handle
    properties
        problemInterface;
    end
    
    methods(Abstract)
        %算法初始化流程
        initial(obj);
        %算法执行流程
        run(obj);     
    end
    methods
        function setProblemInterface(obj,problemInterface)  
            obj.problemInterface = problemInterface;
        end
    end
    
end

