function personDataTransferModelSet = loadData( folderPath )
%开始读取目录
subdir=genpath(folderPath);%列出目录下的所有字目录
filedir=regexp(subdir,pathsep, 'split');%把目录转换成一个cell

personNo = 1;

personDataTransferModelSet(length(filedir)-1) = PersonDataTransferModel();
for personIndex = 1:length(filedir)-1
    
    
    data_fn=dir([filedir{personIndex},filesep,'*.mat']);
    
    if(~isempty(data_fn))
        temp = strfind(filedir{personIndex},filesep);
        path = filedir{personIndex};
        personName = path(temp(end)+1:end);
    else
        continue;
    end
    

    
    for blockIndex = length(data_fn):-1:1

        
        dataFileName = data_fn(blockIndex).name;
        blockDataTransferModel = BlockDataTransferModel();
        
        filePath = [filedir{personIndex},filesep,dataFileName];
        temp = load(filePath);
        blockDataTransferModel.name = dataFileName;
        blockDataTransferModel.data = temp.data;
        
        blockDataTransferModelSet(blockIndex) = blockDataTransferModel;
    end
    personDataTransferModel = PersonDataTransferModel();
    personDataTransferModel.name = personName;
    personDataTransferModel.blockDataTransferModelSet = blockDataTransferModelSet;
    
    personDataTransferModelSet(personNo) = personDataTransferModel;
    personNo = personNo + 1;
end

personDataTransferModelSet(personNo:end) = [];
end
