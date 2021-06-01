close all;
clear all;
clc;

%%
p1 = mfilename('fullpath');
i = findstr(p1,filesep);
p1=p1(1:i(end));
cd(p1);
addpath(genpath([pwd,filesep,'..',filesep]));

%加载数据路径
folderPath = [pwd,filesep,'TestData',filesep];

diary('log.txt');
diary on;

%加载数据
personDataTransferModelSet = loadData(folderPath);

%%
%框架及数据初始化
%初始化框架
frameworkInterface = FrameworkImplement();

frameworkInterface.initial();

frameworkInterface.addData(personDataTransferModelSet);

%%
%第一个算法执行过程
%初始化算法
algorithmInterface = AlgorithmImplement();

%向框架内填充算法
frameworkInterface.addAlgorithm(algorithmInterface);

%执行算法求解
frameworkInterface.run();

%得到评分
scoreModel = frameworkInterface.getScore();

%清除算法结果记录(为下一次计算做准备)
% frameworkInterface.clearAlgorithm();

fprintf('识别结果:\n');

disp(scoreModel.resultTable);

diary off
