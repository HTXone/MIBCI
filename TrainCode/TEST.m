function main()
%TRAINTEST 此处显示有关此函数的摘要
%   此处显示详细说明
    import py.Classifier.*;
    obj = TrainTest();
    obj.method = miClassifier(1);
    data = load('usedData');
    tag = 1;
    
    result = obj.method.TTT(data,tag);
    fprintf("%f\n",result);
    
end