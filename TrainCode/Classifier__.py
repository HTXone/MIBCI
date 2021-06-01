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

import csv

import math


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


class VGG16(nn.Module):
    def __init__(self,channel=1):
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

class CDataSet(Dataset):
    def __init__(self,data,labs,transforms = None,target_transform=None, loader=None):
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
        if mode == 'test':
            self.clf = joblib.load(path)
        else:
            if torch.cuda.is_available():
                self.model = VGG16().cuda()
                #print("CUDA")
            else:
                self.model = VGG16()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.sum = 120

    #==========================================
    '''原接口方法 暂时废弃
    def feature_extraction(self, data):
        fdata = filtfilt(self.BB, self.AA, data)  #滤波
        [_, PxxC3] = scipy_signal.welch(fdata[0, :], fs=1000, nperseg=500, noverlap=250, nfft=2000,return_onesided=True)
        [_, PxxC4] = scipy_signal.welch(fdata[1, :], fs=1000, nperseg=500, noverlap=250, nfft=2000,return_onesided=True)
        Fea = PxxC3 - PxxC4
        #Fea = np.squeeze(Fea)
        return Fea

    def predict(self, fe):
        return self.clf.predict(fe)

    def recognize(self,data):
        fe = self.feature_extraction(data)
        fe = np.array([fe])             # normally the fe should be 2d array
        res = self.clf.predict(fe)
        return res
    '''
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

    '''小波变换创建能量图像'''
    def MakeTrainImgs(self, data, labs):
        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)

        csp.fit(data, labs)

        X = csp.transform(data)

        t = np.arange(0, 4.0, 4.0 / 1000)

        wavename = 'cgau8'
        totalscal = 30
        fc = pywt.central_frequency(wavename)
        cparam = 3 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        for i in (50, 80, 100):
            for j in range(64):
                [cwtmatr, frequencies] = pywt.cwt(X[i][j], scales, wavename, 4.0 / 1000)
                plt.figure(figsize=(8, 8), dpi=100)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.contourf(t, frequencies, abs(cwtmatr))
                # plt.ylabel("Hz")
                # plt.xlabel("S")
                plt.subplots_adjust(hspace=0.4)
                # print(frequencies.shape, cwtmatr.shape, Model.Tdata[0][0].shape)
                if (labs[i] == 1.0):
                    plt.savefig(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/left/%d_%d' % (
                            i, j))
                else:
                    plt.savefig(
                        'D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/right/%d_%d' % (
                            i, j))
                # plt.show()
                # plt.pause(1)

    '''小波变换图像格式变换'''
    def ChangeTrainImgs(self, path):
        Imags = os.listdir(path)
        for i in Imags:
            infile = path + '/' + i
            outfile = path + '/' + i
            im = Image.open(infile)
            (x, y) = im.size  # read image size
            x_s = 64  # define standard width
            y_s = 64  # calc height based on standard width
            out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
            out.save(outfile)

    '''获取小波变换能量图像类型数据'''
    def GetTrainImgs(self,
                     leftPath="D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/left",
                     rightPath="D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/right"):
        Data = np.zeros((120, 64, 64, 64))
        Labs = np.zeros((120))

        LeftImags = os.listdir(leftPath)
        for j in range(0, 60):
            for i in range(0, 64):
                infile = leftPath + "/" + LeftImags[j * 64 + i]
                infile = Image.open(infile)
                infile = infile.convert('L')
                A = np.array(infile)
                # print(A.shape,LeftImags[j*64+i])
                Data[j][i] = A
            Labs[j] = 1
        RightImags = os.listdir(rightPath)
        for j in range(0, 60):
            for i in range(0, 64):
                infile = rightPath + "/" + RightImags[j * 64 + i]
                infile = Image.open(infile)
                infile = infile.convert('L')
                A = np.array(infile)
                # print(A.shape,RightImags[j*64+i])
                Data[j + 60][i] = A
            Labs[j + 60] = 2

        return Data, Labs

    '''获取原csv数据集数据'''
    def GetCsvData(self, path,mode = 'train'):
        if(mode == 'train'):
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
                                                         int(Tindex[(k - 1) * 40 + j]):int(
                                                             Tindex[(k - 1) * 40 + j]) + 1000]
                        x += 1
                    print(Tdata)
            return Tdata, Tlabs
        else :
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
                                                         int(Tindex[(k - 1) * 40 + j]):int(
                                                             Tindex[(k - 1) * 40 + j]) + 1000]
                        x += 1
                    print(Tdata)
            return Tdata, Tlabs

    #=========================================
    '''滤波方法'''

    '''CSP+小波变换 输出Tdata（120*64*64*64）Tlabs（120）'''
    def WavChange(self,data,labs):
        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)

        csp.fit(data, labs)

        X = csp.transform(data)

        print(X.shape)

        #plt.plot(X[0])
        #plt.title("CSP")
        #plt.show()

        w = pywt.Wavelet('db8')
        maxlev = pywt.dwt_max_level(1000, w.dec_len)
        threshold = 0.04

        Y = X
        Y = np.array(Y)
        K = np.zeros((120,64,64))
        print(Y[0].ndim)

        widths = np.arange(1, 65)

        for i in range(120):
            for j in range(64):
                _, K[i][j] = pywt.cwt(Y[i][j], widths, 'mexh')

        return K

    '''CSP+FFT 输出Tdata（1800*64*64） Tlabs（120）'''
    def WavChange_(self,data,labs):
        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)

        DD = csp.fit_transform(data, labs)
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

        time_ = range(0,64)

        plt.plot(XD[0][16])
        plt.title("CSP+FFT")
        plt.show()

        return XD,TL

    '''数据加载器'''
    def DataloderGet(self):

    # np.save('TrainDATA.npy',self.Tdata)
    # np.save('TrainLibs.npy',self.Tlabs)
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

        DataSets = {'Train':CDataSet(self.Tdata[:-150],self.Tlabs[:-150],transforms = transforms.ToTensor),
                    'Test' : CDataSet(self.Tdata[-150:],self.Tlabs[-150:],transforms = transforms.ToTensor)
        }


        dataloders = {x : DataLoader(DataSets[x],batch_size=10,shuffle=True) for x in ['Train','Test']}

        dataset_size = {x : len(DataSets[x]) for x in ['Train', 'Test']}

        return dataloders


    def ModelTest(self,dataloaders):
        epochs = 0  # 内部循环使用
        loss_sum = 0
        acc_sum = 0
        for data, label in dataloaders['Test']:
            if torch.cuda.is_available():
                data = data.float().cuda()
                label = label.long().cuda() - 1
            else:
                data = data
                label = label - 1

            out = self.model(data)

            loss = self.criterion(out, label)
            loss = loss.float()

            loss_sum += loss.data.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred.long() == label.long()).sum()  # .long()
            acc_sum += num_correct.data.item()

            epochs += 1
            train_loss = loss_sum / (epochs)
            train_acc = acc_sum / (epochs)

        print('test loss: {:.4},acc: {:.4}'.format(epochs, train_loss, train_acc))

#==================================================================================
    def ModelTrain(self,dataloaders,num_epochs = 1500):
        epochs = 0  #内部循环使用
        print("迭代次数: ", num_epochs)
        loss_sum = 0
        acc_sum = 0
        for epoch in range(num_epochs):
            for data,label in dataloaders['Train']:
                if torch.cuda.is_available():
                    data = data.float().cuda()
                    label = label.long().cuda()-1
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
                print('steps: {}, loss: {:.4},acc: {:.4}'.format(epochs, train_loss, train_acc))
                self.ModelTest(dataloaders)
#==================================================================================

class TestClassifier:
    def __init__(self):
        self.Q = 0
        self.P = 0
        self.T = 0

    def testScore(self,Q,T,P):
        ITR = 60*(math.log2(Q)+P*math.log2(P)+(1-P)*math.log2((1-P/Q-1)))/T
        return ITR

    def GetCsvData(self, path, mode='train'):
        if (mode == 'train'):
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
                                                         int(Tindex[(k - 1) * 40 + j]):int(
                                                             Tindex[(k - 1) * 40 + j]) + 1000]
                        x += 1
                    print(Tdata)
            self.Data = Tdata
            self.Labs = Tlabs
            return Tdata, Tlabs
        else:
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
                                                         int(Tindex[(k - 1) * 40 + j]):int(
                                                             Tindex[(k - 1) * 40 + j]) + 1000]
                        x += 1
                    print(Tdata)
            self.Data = Tdata
            self.Labs = Tlabs
            return Tdata, Tlabs

    def GetData(self,index = 16,isMore = False):                #index 需要数据包个数
        testData = []
        testLab = 0

        self.randomIndex = np.int(np.random.uniform(0, 120))
        print(self.randomIndex)

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
        self.Q+=1
        self.P+=result

    def WavChange_(self,data,labs):
        csp = CSP(n_components=64, reg=None, log=None, norm_trace=False, transform_into="csp_space")

        # print(Model.Tdata[0].shape)
        DD = data.reshape((1,64,-1))
        # DD = csp.transform(data, labs)
        # print(DD.shape)

        # XL = DD
        TD = np.zeros((1, 64, 384))
        for j in range(64):
            TD[0][j] = fft.fft(DD[0][j], n=384)

        print(TD.shape)
        TD = TD.reshape((1, 64, 6, 64))
        TL = np.zeros((6))
        XD = np.zeros((6, 64, 64))
        for i in range(0, 1):
            for j in range(0, 6):
                for k in range(0, 64):
                    XD[i*6 + j][k] = TD[i][k][j]
                TL[i*6+j] = labs[i]
        print(XD.shape)
        print(TL.shape)

        time_ = range(0,64)

        return XD,TL

    def GetTestData(self):
        self.GetCsvData("D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S01")
        D,L = self.GetData()
        D,L = self.WavChange_(D,L)
        return D,L

if __name__ == '__main__' :
    # Model = miClassifier()
    # DL = Model.DataloderGet()
    # Model.ModelTrain(DL)
    #Model.GetTrainImgs('','')
    #Model.ChangeTrainImgs("D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/right")
    #Model.ChangeTrainImgs("D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/images/left")

    #data= Model.Tdata = np.load("TrainDATA.npy")
    #labs = Model.Tlabs = np.load("TrainLib.npy")

    #Model.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S01')
    #Model.GetCsvData('D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TestCode/TestData/S1',4,6)

    #Model.MakeTrainImgs(data,labs)
    # Data = np.load("testData.npy")
    # print(Data.size)

    Test = TestClassifier()
    Test.GetCsvData("D:/Data/2020-bci-competition-mi-training-local-source-pcode-python/TrainCode/TrainData/S01")
    D,L = Test.GetData()
    Test.WavChange_(D,L)

    # T = [[1,2,3],[4,5,6]]
    # T2 = [[1,2,3],[4,5,6]]
    #
    # T = np.array(T)
    # T2 = np.array(T2)
    #
    # T3 = np.hstack((T,T2))
    #
    # print(T3)



