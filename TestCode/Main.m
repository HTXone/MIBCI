close all;
clear all;
clc;

%%
p1 = mfilename('fullpath');
i = findstr(p1,filesep);
p1=p1(1:i(end));
cd(p1);
addpath(genpath([pwd,filesep,'..',filesep]));

%��������·��
folderPath = [pwd,filesep,'TestData',filesep];

diary('log.txt');
diary on;

%��������
personDataTransferModelSet = loadData(folderPath);

%%
%��ܼ����ݳ�ʼ��
%��ʼ�����
frameworkInterface = FrameworkImplement();

frameworkInterface.initial();

frameworkInterface.addData(personDataTransferModelSet);

%%
%��һ���㷨ִ�й���
%��ʼ���㷨
algorithmInterface = AlgorithmImplement();

%����������㷨
frameworkInterface.addAlgorithm(algorithmInterface);

%ִ���㷨���
frameworkInterface.run();

%�õ�����
scoreModel = frameworkInterface.getScore();

%����㷨�����¼(Ϊ��һ�μ�����׼��)
% frameworkInterface.clearAlgorithm();

fprintf('ʶ����:\n');

disp(scoreModel.resultTable);

diary off
