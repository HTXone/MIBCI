function main()
%TRAINTEST �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    import py.Classifier.*;
    obj = TrainTest();
    obj.method = miClassifier(1);
    data = load('usedData');
    tag = 1;
    
    result = obj.method.TTT(data,tag);
    fprintf("%f\n",result);
    
end