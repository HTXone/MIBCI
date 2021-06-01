# !/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
# @Author �? mrtang
# @Time   :  2019/04/02

import numpy as np
import scipy.signal as scipy_signal
from scipy.signal import filtfilt
from sklearn.externals import joblib
from sklearn.svm import SVC
import os

import xgboost
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from xgboost.sklearn import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
from xgboost import plot_importance

rootdir = os.path.dirname(os.path.abspath(__file__))

class miClassifier:
    def __init__(self,currentPersonId,path=os.path.join(rootdir, r'classifier_params'),mode = 'train'):
        modelName = r'my_model'+ str(int(currentPersonId)) + r'.m'
        path = os.path.join(path,modelName)
        self.BB, self.AA = scipy_signal.butter(2, [0.014, 0.06], btype='bandpass')
        self.personId = currentPersonId
        self.Tdata = None
        self.sum = 0
        self.Tlib = np.array([])
        if mode == 'test':
            self.clf = joblib.load(path)
        else:
            params = {'n_estimators': 150, 'min_child_weight': 3, 'max_depth':6}
            self.clf = xgboost.XGBClassifier(**params)    #SVC(gamma='auto', probability=True,break_ties=False)

    def feature_extraction(self, data):
        fdata = filtfilt(self.BB, self.AA, data)  #滤波
        [_, PxxC3] = scipy_signal.welch(fdata[0, :], fs=1000, nperseg=500, noverlap=250, nfft=2000,return_onesided=True)
        [_, PxxC4] = scipy_signal.welch(fdata[1, :], fs=1000, nperseg=500, noverlap=250, nfft=2000,return_onesided=True)
        Fea = PxxC3 - PxxC4
        Fea = np.squeeze(Fea)
        return Fea[16:30]

    def predict(self, fe):
        return self.clf.predict(fe)

    def recognize(self,data):
        fe = self.feature_extraction(data)
        fe = np.array([fe])             # normally the fe should be 2d array
        res = self.clf.predict(fe)
        return res

    def TTT(self,data,tags):

        self.Tdata = self.Tdata.reshape((self.sum,64,-1))

        np.save('TrainDATA.npy',self.Tdata)
        np.save('TrainLib.npy',self.Tlib)

        fes = None
        fe = None
        Fk = 30-16      #参数

        for i in range(0,self.sum):
            fe = np.squeeze(self.feature_extraction(self.Tdata[i]))
            if i>0:
                fes = np.append(fes,fe)
            else:
                fes = np.array(fe,np.float16)

        fes = fes.reshape((self.sum,-1))    #滤波结果集

        self.clf.fit(fes,self.Tlib)
        '''fes = self.feature_extraction(data)
        fes = np.array([fes])  # normally the fe should be 2d array
        self.clf.fit(fes, tags)
        #'''


    def GetTrainData(self,data,tag):

        if self.sum < 1:
            self.Tdata = np.array(data)
            self.sum =1
        else :
            data = np.array(data, np.float32)
            self.Tdata = np.append(self.Tdata,data)
            self.sum+=1

        self.Tlib = np.append(self.Tlib,tag)

    def save_model(self,path):
        joblib.dump(self.clf, path)

def main():

    #测试用
    Model = miClassifier(1);

    D = np.load('TrainDATA.npy')
    #print(D)
    #D=Model.feature_extraction(D)
    #print(D[0])
    fe = np.squeeze(Model.feature_extraction(D[0]))
    print (fe)
    #D_train = xgboost.DMatrix(X_train,Y_train)
    #X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.5, random_state=0)


    #Model.clf.fit(X_train,Y_train)

   # Y_pred = Model.clf.predict(X_test)
   # print(Y_test,Y_pred)


if __name__ == '__main__' :
    main()

