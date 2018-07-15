# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 07
adaboost.py
"""

import os
import sys
import re
import matplotlib.pyplot as plt
#import shutil
import logging
import numpy as np
import operator
import math
from math import log
from imp import reload
import matplotlib.pyplot as plt
from time import sleep
# Global variable definition here
debug =True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
handler = logging.FileHandler('trace.log',mode='w')
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)
# create a logging format
formatterconsole = logging.Formatter('%(message)s')
handlerconsole = logging.StreamHandler()
handlerconsole.setFormatter(formatterconsole)
handlerconsole.setLevel(logging.INFO)
logger.addHandler(handlerconsole)
# Class definition here

# Function definition here

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    logger.debug("%s = %d", "numFeat", numFeat)
    dataMat = []; labelMat = []
    fr = open(fileName)
    with open(fileName) as fr:
        for line in fr:
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
                dataMat.append(lineArr)
                labelMat.append(float(curLine[-1]))
        return dataMat,labelMat


def loadSimpData():
    """Generate trainning data"""
    datMat = np.matrix([[ 1. , 2.1],
                        [ 2. , 1.1],
                        [ 1.3, 1. ],
                        [ 1. , 1. ],
                        [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                logger.debug('errArr= %s',str(errArr.T))
                logger.debug('predictedVals= %s',str(predictedVals.T))
                #caculate the weighted error
                weightedError = D.T*errArr
                logger.debug("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f",i,threshVal, inequal, weightedError)
#                print ("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" %\
#                                (i, threshVal, inequal, weightedError))
                if weightedError < minError:    
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def plot2D(datMat, labelMat, showWeights=True):
    import matplotlib.pyplot as plt
    #weights = wei.getA()        #.getA(): get array from matrix
    dataArr = np.array(datMat)
    n = np.shape(dataArr)[0]
    #label:1
    xcord1 = [];  ycord1 = []
    #label:-1
    xcord2 = [];  ycord2 = []
    
    xcordSV =[]; ycordSV = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    #for j in range(len(supVector)):
    #    xcordSV.append(supVector[j][0])
    #    ycordSV.append(supVector[j][1])
    #weights = wei.getA()        #.getA(): get array from matrix


    #weights = wei        #.getA(): get array from matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if showWeights == True:
        ax.scatter(xcordSV, ycordSV, s=100, marker='x',c='blue')
        x = np.arange(2.0, 6.0, 0.1)
        #y = (-c-ax)/b
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
    #x = np.arange(-3.0, 3.0, 0.1)
    #y = (-weights[0]-weights[1]*x)/weights[2]
    #ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

#DS means Decision Stump
def adaBoostTrainDS(dataArr,classLabels,numIt=40):      
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)    #increase D for error item in recursion
    aggClassEst = np.mat(np.zeros((m,1)))
    num = 0
    for i in range(numIt):
        logger.info ("D=exp(f(error, alpha))=%s",str(D.T))
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))     #divide a non zero by "max(error,1e-16)"
        logger.info('alpha=log(1-e/e)=%.3f', alpha)
        aggClassEst += alpha*classEst
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        logger.debug("classEst: %s",str(classEst.T))
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)        #wrong: * (1-e)/e, right: * e/1-e
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        #the final result is calc by aggClassEst
        
        num+=1
        logger.debug ("weakClassArr(%d): %s ", num, str(weakClassArr))
        logger.debug ("aggClassEst(%d)=aggClassEst += alpha*classEst: %s ", num, str(aggClassEst.T))
        aggErrors = np.multiply(np.sign(aggClassEst) !=np.mat(classLabels).T, np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        logger.debug ("total error: %d",errorRate)
        if errorRate == 0.0: break
    #return weakClassArr
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                                        classifierArr[i]['thresh'],\
                                                        classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print (aggClassEst)
    return np.sign(aggClassEst)

# Test functions
def test_buildStump():
    D= np.mat(np.ones((5,1))/5)
    datMat, classLabels = loadSimpData()
    #print(buildStump(datMat, classLabels, D))
    plot2D(datMat, classLabels, showWeights=False)
def test_adaBoostTrainDS():
    datMat, classLabels = loadSimpData()
    classifierArray,aggClassEst = adaBoostTrainDS(datMat,classLabels,9)
    #print(classifierArray)
    logger.debug('%s is %s','classifierArray',str(classifierArray))
    
def test_adaClassify():
    datMat, classLabels = loadSimpData()
    classifierArr,aggClassEst = adaBoostTrainDS(datMat,classLabels,9)
    logger.debug("%s = %s",'classifierArr',classifierArr)
    datToClass = np.array([[1.3, 1.5],[5,5],[0,0]])
    result = adaClassify(datToClass,classifierArr)
    logger.debug("%s = %s", 'result.T', result.T)

def test_adaClassifyHorse(num=10, totalResult=None):
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,num)
    errNum, errRate = adaClassifyHorseVerify(datArr, labelArr, classifierArray)
    trainRlt = {};testRlt={}
    trainRlt[num]=[errNum, errRate]
    if totalResult is None:
        totalResult={}
        totalResult['train']={num:None}
        totalResult['test']={num:None}
                
    totalResult['train'].update(trainRlt)
    logger.info("%s %d %.3f","errNum of trainning set is", errNum, errRate)
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    errNum, errRate = adaClassifyHorseVerify(testArr, testLabelArr, classifierArray)
    testRlt[num]=[errNum, errRate]
    totalResult['test'].update(testRlt)
    logger.info("%s %d %.3f","errNum of test set is", errNum, errRate)
     
    
def adaClassifyHorseVerify(datArr, labelArr, classifierArray):
    #testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(datArr,classifierArray)
    n = len(labelArr)
    errArr = np.mat(np.ones((n,1)))
    errArr[prediction10==np.mat(labelArr).T] = 0
    errNum = errArr.sum()
    logger.info("%s %d",'errNum=',errNum)
    errRate = errNum/n
    logger.info("%s %.3f",'errRate=',errRate)
    return errNum, errRate
def test_loopNumofClassifer(start=10, end=60, step=10):
    totalerrNum=[]; totalerrRate=[]
    totalResult = {}
    totalResult['train']={start:None}
    totalResult['test']={start:None}
    for num in range(start, end, step):
        test_adaClassifyHorse(num, totalResult)
    logger.info("total result is %s",str(totalResult))    

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')   #ax.plot([a,b],[c,d],c='b'): draw a line from(a,c) to (b,d) 
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    logger.debug("the Area Under the Curve is: %.2s",str(ySum*xStep))

def test_plotROC():
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst =adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)
def main():
    pass
def removehandle():
    logger.removeHandler(handler)
    logger.removeHandler(handlerconsole)
    

if __name__=='__main__':
    main()
    #test_buildStump()
    #test_adaBoostTrainDS()
    
    #test_adaClassify()
    #test_adaClassifyHorse()
    #test_loopNumofClassifer()
    test_plotROC()
    removehandle()