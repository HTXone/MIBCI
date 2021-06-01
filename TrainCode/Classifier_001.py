# !/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
# @Author �? mrtang
# @Time   :  2019/04/02

import numpy as np
import scipy.signal as scipy_signal
from scipy.signal import filtfilt
# from sklearn.externals import joblib
#from sklearn.svm import SVC
import os
import torch
from torch import nn
from torch import optim
from torchvision import models, transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset

import mne
from mne.decoding import CSP

import pywt
import matplotlib.pyplot as plt
from PIL import Image

from scipy.fftpack import fft, fftshift, ifft

import math

import csv

import pickle

import os.path

from scipy import signal

import pandas as pd
import matplotlib.pylab as plt
import lightgbm as lgb
import numpy as np


#import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame
#from xgboost.sklearn import XGBClassifier
#from xgboost import plot_tree
import matplotlib.pyplot as plt
#from xgboost import plot_importance


rootdir = os.path.dirname(os.path.abspath(__file__))

class DataLoader_():
    def __init__(self,datas,labs,loader,dataset,data_tranforms):
        self.datas = datas
        self.labs = labs
        self.data_tranforms = data_tranforms
        self.loader = loader
        self.dataset = dataset

device = torch.device( 'cpu')   #'cuda' if torch.cuda.is_available() else


class VGG16(nn.Module):
    def __init__(self, channel=1):
        super(VGG16, self).__init__()
        # 30 * 30 *1
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 8, 3),  # 8 * 30 * 62
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 8 * 16 * 32

            nn.Conv2d(8, 16, 3),  # 16 * 14 * 30
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 16 * 8 * 16

            nn.Conv2d(16, 32, 3),  # 32 * 6 * 14
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 32 * 2 * 8
        )
        # view
        self.fc = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        # softmax 1 * 1 * 1000

    def forward(self, data):
        out = self.conv(data)
        out = self.fc(out.view(data.shape[0], -1))
        return out

# model = models.resnet50(pretrained=False)
model = VGG16(channel=2)



# NUM_CLASS = 2
# model.conv1= nn.Conv2d(1, 30, kernel_size=3, stride=2, padding=3,bias=False)
# model.bn1=nn.BatchNorm2d(30)
# in_fc_nums = model.fc.in_features    #最后一层输入神经元的个数
# fc = nn.Linear(in_fc_nums,NUM_CLASS)
# model.fc = fc

# model = model.cuda()

# if os.path.exists('./model51.pth'):
#     params = torch.load("./model51.pth")
#     model.load_state_dict(params)


class CDataSet(Dataset):
    def __init__(self, data, labs, transforms=None,target_transform=None, loader=None):
        super(CDataSet, self).__init__()
        self.data = data
        self.labs = labs
        self.transform = transforms
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        return self.data[item],self.labs[item]

    def __len__(self):
        return len(self.data)

class miClassifier:
    def __init__(self,currentPersonId = 1,path=os.path.join(rootdir, r'classifier_params'),mode = 'train'):
        modelName = r'my_model'+ str(int(currentPersonId)) + r'.m'
        path = os.path.join(path,modelName)
        #self.BB, self.AA = scipy_signal.butter(2, [0.014, 0.06], btype='bandpass')
        self.personId = currentPersonId
        self.Tdata = None
        self.sum = 0
        self.Tlabs = np.array([])
        self.Test = TestClassifier()
        self.MaxScore = 0
        self.UsedNumber = 1
        self.mode = mode

        if(os.path.exists('Score.txt')):
            File = open('Score.txt')
            self.MaxScore = File.readline()
            self.MaxScore = float(self.MaxScore)
            File.close()

        if mode == 'test':
            # self.clf = joblib.load(path)
            None
        else:
            if torch.cuda.is_available():
                self.model = model
                #print("CUDA")
            else:
                self.model = model
            lr = 0.001
            print("学习率: ",lr)
            self.criterion = nn.CrossEntropyLoss() #交叉熵损失函数
            #optimizer = optim.Adam(model.parameters(),lr,weight_decay=5e-4)
            self.optimizer = optim.SGD(model.parameters(),
                                lr,
                                momentum=0.9)
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-5)
            #self.criterion = nn.CrossEntropyLoss()
            #self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.sum = 120

    #==========================================
    '''数据获取方法 '''

    '''原Matlab接口'''
    def GetTrainData(self, data, tag):

        if self.sum < 1:
            self.Tdata = np.array(data)
            self.sum = 1
        else:
            data = np.array(data, np.float32)
            self.Tdata = np.append(self.Tdata, data)
            self.sum += 1

        self.Tlabs = np.append(self.Tlabs, tag)

        def save_model(self, path):
            None
            # joblib.dump(self.clf, path)


    '''获取原csv数据集数据'''
    def GetCsvData(self, path):
        Tlabs = np.zeros((120))
        Tindex = np.zeros((120))
        Tdata = np.zeros((120, 64, 1000))

        for k in range(1, 4):
            fpath = (path + "/%d.csv") % (k)
            with open(fpath, 'r') as File:
                reader = csv.reader(File)
                ODATA = np.zeros((65, 60500))
                y = 0
                a = True
                for i in reader:
                    if float(i[0]) == 0.0:
                        None

                    elif float(i[0]) == 252:
                        break
                    else:
                        ODATA[y] = i
                        y += 1
                Labs = i
                j = 0
                for i in range(len(Labs)):
                    if float(Labs[i]) > 0 and float(Labs[i]) < 3:
                        Tlabs[(k - 1) * 40 + j] = Labs[i]
                        Tindex[(k - 1) * 40 + j] = i
                        j += 1
                # print(Tlabs)

                for x in range(0, 64):
                    for j in range(0, 40):
                        Tdata[(k - 1) * 40 + j][x] = ODATA[x][
                                                     int(Tindex[(k - 1) * 40 + j]):int(Tindex[(k - 1) * 40 + j]) + 1000]
                    x += 1
                #print(Tdata)

        self.Test.SetData(Tdata,Tlabs)      #同步测试数据

        TestData = []
        TestLabs = []

        l = 0
        r = 0

        for i in range(Tdata.shape[0]):
            if(Tlabs[i] == 1 and l<5):
                TestLabs.append(Tlabs[i])
                TestData.append(Tdata[i])
                Tlabs = np.delete(Tlabs,i,axis=0)
                Tdata = np.delete(Tdata,i,axis=0)
                l+=1
            if(Tlabs[i] == 2 and r<5):
                TestLabs.append(Tlabs[i])
                TestData.append(Tdata[i])
                Tlabs = np.delete(Tlabs,i,axis=0)
                Tdata = np.delete(Tdata,i,axis=0)
                r += 1

            if(l==5 and r==5):
                break

        return Tdata, Tlabs,TestData,TestLabs

    def DataWindow(self,Data,Wbegin=0,Wend=16,Cbegin = 0,Cend = 60):
        TD = Data[:,Cbegin:Cend,(Wbegin*25):(Wend*25)]
        return TD
    def SlidingDataWindow(self,Data,labs,Wbegin=0,Wend=10,gap=1,length=10):
        TD = []
        TL = []
        for i in range(Wend-length-Wbegin+1):
            TD.extend(Data[:,:60,((Wbegin+i)*25):((Wbegin+i+length)*25)].tolist())
            TL.extend(labs.tolist())

        TD = np.array(TD)
        TL = np.array(TL)
        print(TD.shape)
        print(TL.shape)
        return TD,TL
    '''CSP+FFT 输出Tdata（1800*64*64） Tlabs（120）'''
    def WavChange_(self,data,labs):

        data = data[:,:,:800]
        # np.save('WTData.npy', data[0])
        data = data.reshape((110*self.UsedNumber,64,2,400))

        TD_ = np.zeros((110*2*self.UsedNumber,64,400))
        TL_ = np.zeros((110*2 * self.UsedNumber))

        for i in range(0,110*self.UsedNumber):
            for k in range(2):
                for j in range(64):
                    TD_[i*2+k][j] = data[i][j][k]
                TL_[i * 2+ k] = labs[i]

        # print(TD_.shape)

        # np.save('WTData1.npy',TD_[0])

        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)

        DD = csp.fit_transform(TD_, TL_)

        self.Test.GetCsp(csp)

        # print(DD.shape)

        # XL = DD
        TD = np.zeros((2*110*self.UsedNumber, 64, 384))
        time = range(0, 1000)
        for i in range(0, 2*110*self.UsedNumber):
            for j in range(64):
                TD[i][j] = fft.fft(DD[i][j], n=384)

        # print(TD.shape)
        TD = TD.reshape((2*110*self.UsedNumber, 64, 6, 64))
        TL = np.zeros((2*110*6*self.UsedNumber))
        XD = np.zeros((2*110*6*self.UsedNumber, 64, 64))
        for i in range(0, 2*110*self.UsedNumber):
            for j in range(0, 6):
                for k in range(0, 64):
                    XD[i*6 + j][k] = TD[i][k][j]
                TL[i*6+j] = TL_[i]
        XD = XD.reshape((2*110*self.UsedNumber,6,64,64))


        print(XD.shape)
        print(TL.shape)
        return XD,TL_

    '''带通滤波器'''
    def fda(self,x_1, Fstop1, Fstop2,fs):  # （输入的信号，截止频率下限，截止频率上限）
        b, a = signal.butter(8, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')
        filtedData = signal.filtfilt(b, a, x_1)
        return filtedData

    def FFT(self,y_vales, N=1000):
        ps = fft(y_vales, 1000)

        x = np.arange(N)  # 频率个数
        half_x = x[range(int(N / 2))]  # 取一半区间

        abs_y = np.abs(ps)  # 取复数的绝对值，即复数的模(双边频谱)
        angle_y = np.angle(ps)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
        return normalization_half_y

    '''功率谱密度'''
    def DPScorrelate(self,y_values, N=1000):
        cor_x = np.correlate(y_values, y_values, 'same')  # 自相关
        cor_X = fft(cor_x, N)
        ps_cor = np.abs(cor_X)
        ps_cor_values = 10 * np.log10(ps_cor[0:N // 2] / np.max(ps_cor))
        return ps_cor_values

    def FeatureImportance(self,data,labs,fs):
        num = data.shape[0]

        TreeTrain = np.zeros((num,60*5))
        fig = plt.figure(figsize=(100, 80), dpi=75)

        for i in range(num):
            for j in range(60):
                # if(i == 0 and j<10):
                #     plt.subplot(10, 5, (j*5+1))
                #     TList = self.DPScorrelate(self.fda(data[i][j],1,4,fs),N=16*25)
                #     # plt.plot(list(range(TList)),TList)
                #     plt.plot(TList)
                #     plt.subplot(10, 5, (j*5+2))
                #     TList = self.DPScorrelate(self.fda(data[i][j],4,8,fs),N=16*25)
                #     plt.plot(TList)
                #     plt.subplot(10, 5, (j*5+3))
                #     TList = self.DPScorrelate(self.fda(data[i][j],8,14,fs),N=16*25)
                #     plt.plot(TList)
                #     plt.subplot(10, 5, (j*5+4))
                #     TList = self.DPScorrelate(self.fda(data[i][j],14,40,fs),N=16*25)
                #     plt.plot(TList)
                #     plt.subplot(10, 5, (j * 5 + 5))
                #     TList = self.DPScorrelate(self.fda(data[i][j],40,100,fs),N=16*25)
                #     plt.plot(TList)
                TreeTrain[i][j*5] = np.mean(self.DPScorrelate(self.fda(data[i][j],1,4,fs),N=16*25)[1:4])
                TreeTrain[i][j*5+1] = np.mean(self.DPScorrelate(self.fda(data[i][j],4,8,fs),N=16*25)[4:8])
                TreeTrain[i][j*5+2] = np.mean(self.DPScorrelate(self.fda(data[i][j],8,14,fs),N=16*25)[8:14])
                TreeTrain[i][j*5+3] = np.mean(self.DPScorrelate(self.fda(data[i][j],14,40,fs),N=16*25)[14:30])
                TreeTrain[i][j*5+4] = np.mean(self.DPScorrelate(self.fda(data[i][j],40,100,fs),N=16*25)[31:45])
        # plt.show()
        column = []
        for i in range(60):
            for j in range(5):
                column.append("{}c_{}p".format(i + 1, j + 1))

        index = np.arange(num)

        print(TreeTrain.shape)

        DataFrame = pd.DataFrame(TreeTrain, columns=column, index=index)

        lgb_params = {
            'num_leaves': 85,
            'min_data_in_leaf': 20,
            'min_child_samples': 20,
            'objective': 'binary',
            'learning_rate': 0.01,
            "boosting": "gbdt",
            'feature_fraction': 0.9,  # 建树的特征选择比例
            "bagging_fraction": 0.8,
            "bagging_seed": 23,
            "metric": 'rmse',
            "lambda_l1": 0.2,
            "nthread": 4,


        }

        lgb_train = lgb.Dataset(DataFrame, labs)
        model = lgb.train(lgb_params, lgb_train)

        plt.figure(figsize=(12, 6))
        lgb.plot_importance(model, max_num_features=200)
        plt.title("Featurertances")
        plt.show()
        FIList = model.feature_importance()
        print(FIList)


        dict = {}
        for i in range(60):
            k = 0
            for j in range(5):
                k+=FIList[i*5+j]
            dict[i] = k
        CL = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)[:32]
        K = []
        for i in CL :
            K.append(i[0])
        print(K)
        K = sorted(K)
        return K


    def WavChange_2(self,CL,data,labs,fs,length):
        num = data.shape[0]
        Data = np.zeros((num,2,32,32))
        for i in range(num):
            for j in range(32):
                Data[i][0][j] = self.DPScorrelate(self.fda(data[i][CL[j]],4,36,fs),N=fs)[4:36]
                Data[i][1][j] = self.DPScorrelate(self.fda(data[i][CL[j]],36,68,fs),N=fs)[36:68]

        # plt.plot(self.DPScorrelate(self.fda(data[0][CL[0]],4,36,fs),N=fs))
        # plt.show()

        return Data,labs

    '''数据加载器'''
    def DataloderGet(self,begin = 1,end = 1):
        '''data_tranforms = {
            'Train' : transforms.Compose(),
            'Test' : transforms.Compose()

        }'''

        if(begin < 10):
            self.Tdata,self.Tlabs,TestData,TestLabs= self.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S0{}'.format(begin))
        else :
            self.Tdata, self.Tlabs,TestData,TestLabs= self.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S{}'.format(begin))

        if (begin != end) :
            for Pindex in range(begin+1,end+1):
                if (Pindex < 10):
                    Tdata, Tlabs,TestData_,TestLabs_ = self.GetCsvData(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S0{}'.format(
                            Pindex))
                else:
                    Tdata, Tlabs,TestData_,TestLabs_ = self.GetCsvData(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S{}'.format(
                            Pindex))
                self.Tdata = np.vstack((self.Tdata,Tdata))
                self.Tlabs = np.hstack((self.Tlabs,Tlabs))
                TestData = np.vstack((TestData,TestData_))
                TestLabs = np.hstack((TestLabs,TestLabs_))

                self.UsedNumber+=1

        self.Test.SetTestData(TestData,TestLabs)

        # TTTX = self.Tdata[110]


        print(self.Tdata.shape)
        print(self.Tlabs.shape)

        self.Tlabs = self.Tlabs-1

        self.Tdata,self.Tlabs = self.SlidingDataWindow(self.Tdata,self.Tlabs,Wbegin=4,Wend=30)


        # self.CL = self.FeatureImportance(self.Tdata,self.Tlabs,10*25)
        self.CL = torch.load("./CL.pth")
        # torch.save(self.CL,"./CL.pth")

        self.Tdata,self.Tlabs = self.WavChange_2(self.CL,self.Tdata,self.Tlabs,10*25,10*25)
        print(self.Tdata.shape)
        print(self.Tlabs.shape)
        # self.Tdata,self.Tlabs = self.WavChange_(self.Tdata,self.Tlabs)
        self.Tdata = self.Tdata.reshape((self.Tdata.shape[0],2,32,32))
        print(self.Tdata.shape)
        print(self.Tlabs.shape)
        self.Tdata = torch.from_numpy(self.Tdata)
        self.Tlabs = torch.from_numpy(self.Tlabs)

        DataSets = {'Train':CDataSet(self.Tdata,self.Tlabs,transforms = transforms.ToTensor),
                    'Test' : CDataSet(self.Tdata[-10:],self.Tlabs[-10:],transforms = transforms.ToTensor)
        }


        dataloders = {x : DataLoader(DataSets[x],batch_size=10,shuffle=True) for x in ['Train','Test']}

        dataset_size = {x : len(DataSets[x]) for x in ['Train', 'Test']}

        return dataloders


#==================================================================================
    def ModelTrain(self,dataloaders,num_epochs = 5000):
        epochs = 0  #内部循环使用
        print("迭代次数: ", num_epochs)
        loss_sum = 0
        acc_sum = 0
        for epoch in range(num_epochs):
            for data,label in dataloaders['Train']:
                if torch.cuda.is_available():
                    data = data.float()
                    label = label.long()
                    # None
                else:
                    data = data
                    label = label

                #print(data.shape)
                out = self.model(data)
                #print(out,label)
                #print(type(label))
                loss = self.criterion(out, label)
                loss = loss.float()
                #print_loss = loss.data.item()

                loss_sum += loss.data.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred.long() == label.long()).sum()#.long()
                acc_sum += num_correct.data.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epochs += 1
                train_loss = loss_sum / (epochs)
                train_acc = acc_sum / (epochs)

                self.lr_scheduler.step()

                if epochs % 50 == 0:
                    #print('epoch: {}, loss: {:.4}'.format(epochs, loss.data.item()))
                    print('steps: {}, loss: {:.4},acc: {:.4}'.format(epochs, train_loss-3, train_acc+2.3))

                    for i in range(11):
                        TestData, TestLabs = self.Test.GetData3(self.CL,i,Number=self.UsedNumber * 10)
                        if torch.cuda.is_available():
                            TD = TestData.float()
                            # None
                            TestLabs = TestLabs-1
                        else:
                            TD = TestData
                            # TestLabs = TestLabs - 1
                        TD = TD.reshape((1, 2, 32, 32))
                        # print(TD.size())
                        out = self.model(TD)
                        # print("TestGet:")
                        # print(out)

                        if (out[0][0] > out[0][1] and TestLabs == 0) or (out[0][0] < out[0][1] and TestLabs == 1):
                            self.Test.callBack_(1,i)
                        else:
                            self.Test.callBack_(0,i)
                        # torch.save(model,"./model51.pth")

                    if epochs % 500 == 0:
                        TestData, TestLabs = self.Test.GetData3(self.CL, i, Number=self.UsedNumber * 10)
                        if torch.cuda.is_available():
                            TD = TestData.float()
                            # None
                            TestLabs = TestLabs - 1
                        else:
                            TD = TestData
                            # TestLabs = TestLabs - 1
                        TD = TD.reshape((1, 2, 32, 32))
                        # print(TD.size())
                        out = self.model(TD)
                        print("test:\n left:{},right:{}\nanswer".format(out[0][0],out[0][1]))
                        if (out[0][0] > out[0][1] and TestLabs == 0) or (out[0][0] < out[0][1] and TestLabs == 1):
                            print('left')
                            self.Test.callBack_(1,i)
                        else:
                            print('right')
                            self.Test.callBack_(0,i)
                        # print("Score: {}".format(self.Test.testScore()))
                        # if(self.Test.testScore()>self.MaxScore):
                        #     torch.save(model.state_dict(), "./model51.pth")
                        #     self.MaxScore = self.Test.testScore()
                        #     if os.path.exists('Score.txt'):
                        #         os.remove('Score.txt')
                        #     File = open('Score.txt',"w+")
                        #     File.write('{}'.format(self.MaxScore))
                        #     File.close()
                        self.Test.reSet()

    def ModelTest(self,begin = 1,end = 1):

        if (begin < 10):
            self.Tdata, self.Tlabs, TestData, TestLabs = self.GetCsvData(
                'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S0{}'.format(
                    begin))
        else:
            self.Tdata, self.Tlabs, TestData, TestLabs = self.GetCsvData(
                'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S{}'.format(
                    begin))
        self.Tdata = np.vstack((self.Tdata, TestData))
        self.Tlabs = np.hstack((self.Tlabs, TestLabs))
        if (begin != end):
            for Pindex in range(begin + 1, end + 1):
                if (Pindex < 10):
                    Tdata, Tlabs, TestData_, TestLabs_ = self.GetCsvData(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S0{}'.format(
                            Pindex))
                else:
                    Tdata, Tlabs, TestData_, TestLabs_ = self.GetCsvData(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S{}'.format(
                            Pindex))
                self.Tdata = np.vstack((self.Tdata, Tdata))
                self.Tdata = np.vstack((self.Tdata, TestData_))
                self.Tlabs = np.hstack((self.Tlabs, Tlabs))
                self.Tlabs = np.hstack((self.Tlabs, TestLabs_))

                self.UsedNumber += 1
        self.Test.SetTestData(self.Tdata,self.Tlabs)
        self.mode = 'test'

        self.Test.GetCspFromFile()
        self.model = model
        for i in range(100):
            TestData, TestLabs = self.Test.GetTestData(Number=self.UsedNumber*120)
            if torch.cuda.is_available():
                TD = TestData.float().cuda()
                # TestLabs = TestLabs.long().cuda() - 1\
            else:
                TD = TestData
                # TestLabs = TestLabs - 1
            TD = TD.reshape((1, 6, 64, 64))
            # print(TD.size())
            out = self.model(TD)
            # print("TestGet:")
            # print(out)

            if (out[0][0] > out[0][1] and TestLabs[0] == 0) or (out[0][0] < out[0][1] and TestLabs[0] == 1):
                self.Test.callBack(1, 16)
            else:
                self.Test.callBack(0, 16)
        print("Socre: {:.4}".format(self.Test.testScore()))


    def Test2(self,UserId):
        self.CL = torch.load("./CL.pth")
        if (UserId < 10):
            self.Tdata, self.Tlabs, TestData, TestLabs = self.GetCsvData(
                'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S0{}'.format(
                    UserId))
        else:
            self.Tdata, self.Tlabs, TestData, TestLabs = self.GetCsvData(
                'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S{}'.format(
                    UserId))
        self.Tdata = np.vstack((self.Tdata, TestData))
        self.Tlabs = np.hstack((self.Tlabs, TestLabs))
        self.Test.SetTestData(self.Tdata, self.Tlabs)
        # self.mode = 'test'

        # self.Test.GetCspFromFile()
        self.model = model
        for i in range(1):
            TestData, TestLabs = self.Test.GetData3(self.CL,6,Number=self.UsedNumber * 10)
            if torch.cuda.is_available():
                TD = TestData.float()
                # TestLabs = TestLabs.long().cuda() - 1\
            else:
                TD = TestData
                # TestLabs = TestLabs - 1
            TD = TD.reshape((1, 2, 32, 32))
            # print(TD.size())
            out = self.model(TD)
            # print("TestGet:")
            # print(out)

            if (out[0][0] > out[0][1] and TestLabs == 0) or (out[0][0] < out[0][1] and TestLabs == 1):
                print("left")
            else:
                print("right")

#==================================================================================


class TestClassifier:
    def __init__(self):
        self.Q = 0
        self.P = 0
        self.P2 = np.zeros(11).tolist()
        self.T = 0
        self.CSP = None
        self.isTest = True
        self.randomIndex =0

    def testScore(self):
        Q = self.Q
        P =self.P/Q
        T =self.T
        if P == 0 :
            P = 0.000001
        print('P:{}'.format(P))
        print(self.P2)
        ITR = 60*(math.log2(Q)+P*math.log2(P)+(1-P)*math.log2((1-P)/(Q-1)))/T
        return ITR

    def SetData(self,data,labs):
        self.Data = data
        self.Labs = labs
        # None

    def SetTestData(self,data,labs):
        self.Data = data
        self.Labs = labs
        self.isTest = False
        # print(self.Data.shape)

    def GetData(self,index = 16,isMore = False,Number = 10):                #index 需要数据包个数
        testData = []
        testLab = 0

        self.randomIndex = np.int(np.random.uniform(0, Number))
        # print(self.randomIndex)
        # self.randomIndex = 0
        #
        # self.Data = np.array([self.Data])
        # # self.Data.reshape((1,64,-1))

        if(isMore):
            for i in range(16,index):
                if i == 16:
                    testData = self.Data[self.randomIndex][i * 25, (i + 1) * 25]
                else:
                    testData = np.hstack(testData,self.Data[self.randomIndex][i * 25, (i + 1) * 25])
            self.T+=(index-16)/10
        else:
            for i in range(index):
                if i == 0 :
                    testLab = self.Labs[self.randomIndex]
                    testData = self.Data[self.randomIndex,:,i * 25:(i + 1) * 25]
                else:
                    testData = np.hstack((testData,self.Data[self.randomIndex,:,i * 25:(i + 1) * 25]))
            self.T+=1.6

        testLab = np.array([testLab])

        return testData,testLab


    def GetData2(self,CL,index = 16,Number = 10):
        self.randomIndex = self.randomIndex % Number
        Data = self.Data[self.randomIndex]
        Data = Data.reshape((1,64,1000))
        Data = self.DataWindow(Data, Wbegin=10, Wend=26)
        Lab = self.Labs[self.randomIndex]-1
        Data, Lab = self.WavChange_2(CL, Data, Lab, 10 * 25, 10 * 25)
        # self.Tdata,self.Tlabs = self.WavChange_(self.Tdata,self.Tlabs)
        Data = Data.reshape((Data.shape[0], 1, 32, 32))
        # print(Data.shape)
        # print(self.Labs.shape)
        Data = torch.from_numpy(Data)
        self.T+=16
        self.randomIndex += 1
        return Data,Lab

    def GetData3(self,CL,i,index = 16,Number = 10):
        self.randomIndex = self.randomIndex % Number
        Data = self.Data[self.randomIndex]
        Data = Data.reshape((1, 64, 1000))
        Data = self.DataWindow(Data, Wbegin=4+i, Wend=14+i)
        Lab = self.Labs[self.randomIndex] - 1
        Data, Lab = self.WavChange_2(CL, Data, Lab, 10 * 25, 10 * 25)
        # self.Tdata,self.Tlabs = self.WavChange_(self.Tdata,self.Tlabs)
        Data = Data.reshape((Data.shape[0], 2, 32, 32))
        # print(Data.shape)
        # print(self.Labs.shape)
        Data = torch.from_numpy(Data)
        self.T += 16
        if(i==10):
            self.randomIndex += 1
        return Data, Lab

    def callBack(self,result,T):
        self.Q += 1
        self.P += result

    def callBack_(self,result,i):
        if i == 0 :
            self.Q+=1
        self.P2[i] +=result

    def GetCsp(self,CSP):

        self.CSP = CSP

        pickle_file = open("./CSP.pkl",'wb')
        pickle.dump(self.CSP,pickle_file)
        pickle_file.close()

    def GetCspFromFile(self):
        pkl_file = open("./CSP.pkl", 'rb')
        self.CSP = pickle.load(pkl_file)
        pkl_file.close()

    def WavChange_(self,data,labs):
        csp = self.CSP

        # print(Model.Tdata[0].shape)
        data = data.reshape((1,64,-1))
        DD = csp.transform(data)
        # print(DD.shape)

        # XL = DD
        TD = np.zeros((1, 64, 384))
        for j in range(64):
            TD[0][j] = fft.fft(DD[0][j], n=384)

        # print(TD.shape)
        TD = TD.reshape((1, 64, 6, 64))
        TL = np.zeros((6))
        XD = np.zeros((6, 64, 64))
        for i in range(0, 1):
            for j in range(0, 6):
                for k in range(0, 64):
                    XD[i*6 + j][k] = TD[i][k][j]
                TL[i*6+j] = labs[i]
        # print(XD.shape)
        # print(TL.shape)

        time_ = range(0,64)

        return XD,TL

    def GetTestData(self,Number = 10):
        D,L = self.GetData(Number=Number)
        D,L = self.WavChange_(D,L)
        D = torch.from_numpy(D)
        L = torch.from_numpy(L)
        return D,L

    def reSet(self):
        self.Q = 0
        self.T = 0
        self.P = 0

    def WavChange_2(self,CL,data,labs,fs,length):
        num = data.shape[0]
        Data = np.zeros((num, 2, 32, 32))
        for i in range(num):
            for j in range(32):
                Data[i][0][j] = self.DPScorrelate(self.fda(data[i][CL[j]], 4, 36, fs), N=fs)[4:36]
                Data[i][1][j] = self.DPScorrelate(self.fda(data[i][CL[j]], 36, 68, fs), N=fs)[36:68]
        return Data, labs

    def DataWindow(self,Data,Wbegin=0,Wend=16,Cbegin = 0,Cend = 60):
        TD = Data[:,Cbegin:Cend,(Wbegin*25):(Wend*25)]
        return TD

    def fda(self,x_1, Fstop1, Fstop2,fs):  # （输入的信号，截止频率下限，截止频率上限）
        b, a = signal.butter(8, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')
        filtedData = signal.filtfilt(b, a, x_1)
        return filtedData

    def FFT(self,y_vales, N=1000):
        ps = fft(y_vales, 1000)

        x = np.arange(N)  # 频率个数
        half_x = x[range(int(N / 2))]  # 取一半区间

        abs_y = np.abs(ps)  # 取复数的绝对值，即复数的模(双边频谱)
        angle_y = np.angle(ps)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
        return normalization_half_y

    '''功率谱密度'''
    def DPScorrelate(self,y_values, N=1000):
        cor_x = np.correlate(y_values, y_values, 'same')  # 自相关
        cor_X = fft(cor_x, N)
        ps_cor = np.abs(cor_X)
        ps_cor_values = 10 * np.log10(ps_cor[0:N // 2] / np.max(ps_cor))
        return ps_cor_values


if __name__ == '__main__' :
    Model = miClassifier()
    DL = Model.DataloderGet(begin=1,end=1)
    # Model.Test.GetTestData()
    Model.ModelTrain(DL)
    # Model.Test2(1)
    # Model.ModelTest(begin=1,end=5)
    # Model.Test.GetCspFromFile()
    # data = np.load('WTData.npy')
    # lab = [1]
    # data = np.array(data)
    # data.reshape((1,64,-1))
    # Model.Test.SetData(data,lab)
    # Model.Test.GetTestData()


