# !/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
# @Author �? mrtang
# @Time   :  2019/04/02

import numpy as np
import scipy.signal as scipy_signal
from scipy.signal import filtfilt
from sklearn.externals import joblib
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

from numpy import fft

import math

import csv

import pickle

import os.path


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
class VGG16(nn.Module):
    def __init__(self, channel=1):
        super(VGG16, self).__init__()
        # 64 * 64 *1
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 8, 3),  # 8 * 62 * 62
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 8 * 32 * 32

            nn.Conv2d(8, 16, 3),  # 16 * 30 * 30
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 16 * 16 * 16

            nn.Conv2d(16, 32, 3),  # 32 * 14 * 14
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),  # 32 * 14 * 14
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 32 * 8 * 8

            nn.Conv2d(32, 64, 3),  # 64 * 6 * 6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),  # 64 * 6 * 6
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 64 * 4 * 4

            nn.Conv2d(64, 64, 3),  # 64 * 2 * 2
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),  # 64 * 2 * 2
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 64 * 2 * 2
        )
        # view
        self.fc = nn.Sequential(
            nn.Linear(64*2*2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # softmax 1 * 1 * 1000

    def forward(self, data):
        out = self.conv(data)
        out = self.fc(out.view(data.shape[0], -1))
        return out
'''

if os.path.exists("./model.pkl") :
    model = torch.load("./model.pkl")
else:
    model = models.resnet50(pretrained=True)
    NUM_CLASS = 2
    model.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    in_fc_nums = model.fc.in_features    #最后一层输入神经元的个数
    fc = nn.Linear(in_fc_nums,NUM_CLASS)
    model.fc = fc
    model = model.cuda()


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
        self.BB, self.AA = scipy_signal.butter(2, [0.014, 0.06], btype='bandpass')
        self.personId = currentPersonId
        self.Tdata = None
        self.sum = 0
        self.Tlabs = np.array([])
        self.Test = TestClassifier()

        if mode == 'test':
            self.clf = joblib.load(path)
        else:
            if torch.cuda.is_available():
                self.model = model.cuda()
                #print("CUDA")
            else:
                self.model = model
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
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
            joblib.dump(self.clf, path)


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

        return Tdata, Tlabs


    '''CSP+FFT 输出Tdata（1800*64*64） Tlabs（120）'''
    def WavChange_(self,data,labs):
        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)

        DD = csp.fit_transform(data, labs)

        self.Test.GetCsp(csp)

        print(DD.shape)

        # XL = DD
        TD = np.zeros((120, 64, 960))
        time = range(0, 1000)
        for i in range(0, 120):
            for j in range(64):
                TD[i][j] = fft.fft(DD[i][j], n=960)

        print(TD.shape)
        TD = TD.reshape((120, 64, 15, 64))
        TL = np.zeros((1800))
        XD = np.zeros((1800, 64, 64))
        for i in range(0, 120):
            for j in range(0, 15):
                for k in range(0, 64):
                    XD[i*15 + j][k] = TD[i][k][j]
                TL[i*15+j] = labs[i]
        print(XD.shape)
        print(TL.shape)
        return XD,TL

    '''数据加载器'''
    def DataloderGet(self):
        '''data_tranforms = {
            'Train' : transforms.Compose(),
            'Test' : transforms.Compose()

        }'''
        self.Tdata,self.Tlabs= self.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S01')
        self.Tdata,self.Tlabs = self.WavChange_(self.Tdata,self.Tlabs)
        self.Tdata = self.Tdata.reshape((1800,1,64,64))
        print(self.Tdata.shape)
        self.Tdata = torch.from_numpy(self.Tdata)
        self.Tlabs = torch.from_numpy(self.Tlabs)

        DataSets = {'Train':CDataSet(self.Tdata,self.Tlabs,transforms = transforms.ToTensor),
                    'Test' : CDataSet(self.Tdata[-150:],self.Tlabs[-150:],transforms = transforms.ToTensor)
        }


        dataloders = {x : DataLoader(DataSets[x],batch_size=10,shuffle=True) for x in ['Train','Test']}

        dataset_size = {x : len(DataSets[x]) for x in ['Train', 'Test']}

        return dataloders


#==================================================================================
    def ModelTrain(self,dataloaders,num_epochs = 30):
        epochs = 0  #内部循环使用
        print("迭代次数: ", num_epochs)
        loss_sum = 0
        acc_sum = 0
        for epoch in range(num_epochs):
            for data,label in dataloaders['Train']:
                if torch.cuda.is_available():
                    data = data.float().cuda()
                    label = label.long().cuda() - 1
                else:
                    data = data
                    label = label-1

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

                if epochs % 50 == 0:
                    print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
                    for i in range(10):
                        TestData, TestLabs = self.Test.GetTestData()
                        if torch.cuda.is_available():
                            TD = TestData.float().cuda()
                            # TestLabs = TestLabs.long().cuda() - 1\
                        else:
                            TD = TestData
                            # TestLabs = TestLabs - 1
                        TD = TD.reshape((6, 1, 64, 64))
                        # print(TD.size())
                        out = self.model(TD)
                        # print("TestGet:")
                        # print(out)
                        l = 0
                        r = 0
                        for testI in range(6):
                            if out[testI][0] > out[testI][1]:
                                l += 1
                            else:
                                r += 1
                        if (l > r and TestLabs[0] == 1) or (l < r and TestLabs[0] == 2):
                            self.Test.callBack(1, 6)
                        else:
                            self.Test.callBack(0, 6)
                    torch.save(model,"./model.pkl")
                    if  epochs % 500 == 0:
                        print("Socre: {:.4}".format(self.Test.P))
                        self.Test.reSet()

    def ModelTest(self):
        self.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S01')
        self.Test.GetCspFromFile()
        self.model = torch.load("./model.pkl")
        for i in range(100):
            TestData, TestLabs = self.Test.GetTestData()
            if torch.cuda.is_available():
                TD = TestData.float().cuda()
                # TestLabs = TestLabs.long().cuda() - 1\
            else:
                TD = TestData
                # TestLabs = TestLabs - 1
            TD = TD.reshape((6, 1, 64, 64))
            # print(TD.size())
            out = self.model(TD)
            # print("TestGet:")
            # print(out)
            l = 0
            r = 0
            for testI in range(6):
                if out[testI][0] > out[testI][1]:
                    l += 1
                else:
                    r += 1
            if (l > r and TestLabs[0] == 1) or (l < r and TestLabs[0] == 2):
                self.Test.callBack(1, 16)
            else:
                self.Test.callBack(0, 16)
        print("Socre: {:.4}".format(self.Test.testScore()))

#==================================================================================


class TestClassifier:
    def __init__(self):
        self.Q = 0
        self.P = 0
        self.T = 0
        self.CSP = None

    def testScore(self):
        Q = self.Q
        P =self.P/Q
        T =self.T
        if P == 0 :
            P = 0.000001
        ITR = 60*(math.log2(Q)+P*math.log2(P)+(1-P)*math.log2((1-P)/(Q-1)))/T
        return ITR

    def SetData(self,data,labs):
        self.Data = data
        self.Labs = labs

    def GetData(self,index = 16,isMore = False):                #index 需要数据包个数
        testData = []
        testLab = 0

        self.randomIndex = np.int(np.random.uniform(0, 120))
        # print(self.randomIndex)

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

    def callBack(self,result,T):
        self.Q += 1
        self.P += result

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

    def GetTestData(self):
        D,L = self.GetData()
        D,L = self.WavChange_(D,L)
        D = torch.from_numpy(D)
        L = torch.from_numpy(L)
        return D,L

    def reSet(self):
        self.Q = 0
        self.T = 0
        self.P = 0


if __name__ == '__main__' :
    Model = miClassifier()
    # DL = Model.DataloderGet()
    # Model.Test.GetTestData()
    # Model.ModelTrain(DL)
    Model.ModelTest()
    # Q = 10
    # P = 0.5
    # T = 1
    # ITR = 60 * (math.log2(Q) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (Q - 1))) / T
    # print(ITR)