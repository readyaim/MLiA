# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 05
logRegres.py
"""
import os
import sys
import re
import matplotlib.pyplot as plt
#import shutil
#import logging
import numpy as np
import operator
import math
from math import log
from imp import reload
import matplotlib.pyplot as plt

# Global variable definition here
debug =True

# Class definition here
class Class_definition(object):
    pass
# Function definition here
def loadDataSet():
    dataMat = []; labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):   #inX(100, 1)
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  #(100,3)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)  #100, 3
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #(100,1) = sigmoid((100, 3) * (3, 1))
        error = (labelMat - h)  #error.(100,1)
        weights = weights + alpha * dataMatrix.transpose()* error   #weights(3,1) = alpha*(3,100)*(100,1)
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()        #.getA(): get array from matrix
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    weights_reshape = weights.reshape((n,1))        #to fit the argument of plotBestFit()
    return weights_reshape

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    error_log=[]
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01            #why 4? why 0.01?
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = (classLabels[randIndex] - h)
            weights = weights + alpha * error * dataMatrix[randIndex]   #(1,1)*(1,1)*(1,3)
            del(dataIndex[randIndex])
            #error_log.append(error)
    weights_reshape = weights.reshape((n,1))        #to fit the argument of plotBestFit()
    #print(error_log)
    return weights_reshape

def classifyVector(inX, weights):
    #prob = sigmoid(sum(inX*weights))
    prob = sigmoid(sum(inX*weights.ravel()))
    if prob.all() > 0.5: 
        return 1.0
    else: 
        return 0.0
def colicTest():
    frTrain = open('horseColicTraining.txt') 
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %3f" % errorRate)
    return errorRate

def multiTest_colicTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is:%f" \
                % (numTests, errorSum/float(numTests)))
# Test functions
def test_loadDataSet():
    dataArr,labelMat=loadDataSet()
    weights = gradAscent(dataArr,labelMat)
    print(weights)
    #plotBestFit(weights.getA())
    plotBestFit(weights)

def test_stocGradAscent0():
    dataArr,labelMat=loadDataSet()
    weights=stocGradAscent0(np.array(dataArr),labelMat)
    print(weights)
    plotBestFit(np.mat(weights))

def test_stocGradAscent1():
    dataArr,labelMat=loadDataSet()
    weights=stocGradAscent1(np.array(dataArr),labelMat)
    print(weights)
    plotBestFit(np.mat(weights))

        
def main():
    test_loadDataSet()
    #test_stocGradAscent0()
    #test_stocGradAscent1()
    #colicTest()
    #multiTest_colicTest()
if __name__=='__main__':
    #datingClassTest()
    main()
    