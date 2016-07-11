# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 15:35:32 2016

@author: amulya
"""
import nnCls
import pandas as pd
import numpy as np

#orTable = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
##print orTable
#nn_test = nnCls.nn(orTable[:,:2],orTable[:,2:],3)
#nn_test.train_nn(orTable[:,:2],orTable[:,2:],0.25,501)
#nn_test.conf_matrix(orTable[:,:2],orTable[:,2:])
def norm(input):
    input=(input-input.mean(axis=0))/input.std(axis=0)
    return input
dataFrame = pd.read_csv("iris_data.csv", header=None, names = ["sepal length", "sepal width", "petal length", "petal width","class"])
dataFrame["class_num"] = dataFrame["class"].map({"Iris-setosa":0, "Iris-versicolour":1, "Iris-virginica":2})
#print dataFrame.head(-5)
input_data = dataFrame.iloc[:,:4].values
input_data = norm(input_data)
target = dataFrame.loc[:149,"class_num"].values
clsTot = 3


target0 = np.zeros((target.shape[0], clsTot))
target0[target[:]==0,0]=1 #if col0 = 0 (value of output) , then place 1 in the x,3 matrix for the 0th col
target0[target[:]==1,1]=1
target0[target[:]==2,2]=1

order = range(input_data.shape[0])
np.random.shuffle(order)
input_data=input_data[order,:]
target0=target0[order,:]

#breaking data into train, test and validate sets
trainIp = input_data[::2,:]
trainTarget = target0[::2]
print trainTarget.shape
validateIp = input_data[1::4,:]
validateTarget =  target0[1::4]
testIp = input_data[3::4,:]
testTarget = target0[3::4]

nn_test = nnCls.nn(trainIp,trainTarget,5)
nn_test.early_stop_validate(trainIp,trainTarget,validateIp,validateTarget,0.25)
#nn_test.train_nn(trainIp,trainTarget,0.25,501)
nn_test.conf_matrix(testIp,testTarget)
