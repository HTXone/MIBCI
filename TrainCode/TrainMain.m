function TrainMain()
%TRAINMAIN 此处显示有关此函数的摘要
%   此处显示详细说明
    
    personDataTransferModelSet = DataGet("D:\Data\2020-bci-competition-mi-training-local-source-pcode-python\TrainCode\TrainData");
    [row,cal] = size(personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, 1).data);
    
    fprintf("%d,%d\n",row,cal);
    
    LeftBlockSet = {};
    LeftBlock = [];
    left = 1;
    RightBlockSet = {};
    RightBlock = [];
    right = 1;
    
    a = 0; 
    now = 0;
    
    %fprintf("%d",personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, 1).data(65,1));
    import py.Classifier.*;
    
    TrainModel = miClassifier(1);
    
    for j = 1:3
        for i = 1:cal     
        if(personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, j).data(65,i) == 1)
            a = 1;
            now = 1;
        elseif(personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, j).data(65,i) == 2)
            a=1;
            now=2;
        end
        if(a == 1&&now == 1)
            LeftBlock = [LeftBlock,personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, j).data(1:64,i)];
        elseif(a==1&&now==2)
            RightBlock = [RightBlock,personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, j).data(1:64,i)];
        end
        if(personDataTransferModelSet(1, 1).blockDataTransferModelSet(1, j).data(65,i) == 251)
            if(now == 1)
                %TrainModel.feature_extraction(LeftBlock);
                TrainModel.GetTrainData(LeftBlock,1);
                LeftBlockSet{left} = LeftBlock;
                left = left+1;
                LeftBlock = [];
            elseif(now == 2)
                TrainModel.GetTrainData(RightBlock,2);
                RightBlockSet{right} = RightBlock;
                right = right+1;
                RightBlock = [];
            end
            a = 0;
            now = 0;
        end
        end
    end
    
    result = TrainModel.TTT(LeftBlockSet,1);
    
end

