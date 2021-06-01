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
    def __init__(self,currentPersonId,path=os.path.join(rootdir, r'classifier_params'),mode = 'test'):
        modelName = r'my_model'+ str(int(currentPersonId)) + r'.m'
        path = os.path.join(path,modelName)
        self.BB, self.AA = scipy_signal.butter(2, [0.014, 0.06], btype='bandpass')
        self.personId = currentPersonId
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
        return Fea[16:30]

    def predict(self, fe):
        return self.clf.predict(fe)

    def recognize(self,data):
        print(data.size)
        fe = self.feature_extraction(data)
        fe = np.array([fe])             # normally the fe should be 2d array
        res = self.clf.predict(fe)
        return res

    def getTrainData(self,Ldata,Rdata):
        self.Tdata = Ldata+Rdata;
        Tlib = np.zeros(60)
        Tlib = np.append(Tlib, np.ones(60))
        #print(Tlib)
        self.Tlib = Tlib

    def train(self,data,tags):
        fes = self.feature_extraction(data)
        fes = np.array([fes])  # normally the fe should be 2d array
        #self.clf.fit(fes)
        return self.clf.predict(fes)

    def save_model(self,path):
        joblib.dump(self.clf, path)

def main():
    Model = miClassifier(1);

    X_train = [[[1, 1],[1, 1]], [[1, 1],[1, 1]], [[1, 1],[1, 1]], [[0, 0],[0, 0]],[[0 , 0],[0, 0]],[[0 , 0],[0, 0]]]
    Y_train = [1, 1, 1, 0 , 0,0]
    X_train = np.array(X_train,np.float32)
    Y_train = np.array(Y_train,np.float32)
    #D_train = xgboost.DMatrix(X_train,Y_train)
    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.5, random_state=0)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    Model.clf.fit(X_train,Y_train)

    Y_pred = Model.clf.predict(X_test)
    print(Y_test,Y_pred)


if __name__ == '__main__' :
    main()

