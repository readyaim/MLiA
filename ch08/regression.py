# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 08
regression.py
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
handlerconsole.setLevel(logging.DEBUG)
logger.addHandler(handlerconsole)
#remove handles
def removehandle():
    logger.removeHandler(handler)
    logger.removeHandler(handlerconsole)
# Class definition here

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    #fr = open(fileName)
    with open(fileName) as fr:
        for line in fr:
            lineArr =[]
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def plot2DScatter(xArr, yArr, ws=None,plotEnable=False):
    if plotEnable==True:
        xMat = np.mat(xArr); yMat = np.mat(yArr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])    #ax.scatter([x1,x2],[y1,y2])

        xCopy = np.mat(xArr).copy()
        xCopy.sort(0)   #plot step by step
        yHat = xCopy*ws    
        ax.plot(xCopy[:,1], yHat, color='r')
        #ax.plot([xCopy[0].A[0][1], xCopy[-1].A[0][1]],[yHat[0].A[0][0], yHat[-1].A[0][0]],color='r')
        
        plt.show()        
def plot2DScatterLWLR(xArr, yArr, yHat_lwlr, ws=None,plotEnable=False):
    if plotEnable==True:
        xMat = np.mat(xArr); yMat = np.mat(yArr); yMat_lwlr = np.mat(yHat_lwlr)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])    #ax.scatter([x1,x2],[y1,y2])

        xCopy = np.mat(xArr).copy()
        sortedIndicies = xMat.A[:,1].argsort()  #sorted indice by xMat[:,1]
        xCopy.sort(0)   #plot step by step
        yHat = xCopy*ws    
        ax.plot(xCopy[:,1], yHat, color='r')
        #ax.scatter(xMat[:,1].flatten().A[0], yMat_lwlr.T[:,0].flatten().A[0], color='y')
        #ax.plot(xCopy[:,1], yHat_lwlr, color=(0.1, 0.2, 0.5))
        ax.plot(xMat[:,1].A[sortedIndicies], yHat_lwlr[sortedIndicies], color=(0.1, 0.2, 0.5))

        #ax.plot([xCopy[0].A[0][1], xCopy[-1].A[0][1]],[yHat[0].A[0][0], yHat[-1].A[0][0]],color='r')
        
        plt.show()    
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        logger.warning ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    
    return ws
#local weighted linear regression
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        #no inverse xTx.I
        logger.info("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

# Function definition here
def main():
    pass

# Test functions
def test_standRegres():
    xArr,yArr=loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr)
    logger.debug("ws=%s",str(ws))
    plot2DScatter(xArr, yArr,ws, True)
    yHat= xArr * ws
    corr_coeaf = np.corrcoef(yHat.T, yArr)
    logger.debug("corr_coeaf is %s", corr_coeaf)
    return 
    
def test_lwlr():
    xArr,yArr=loadDataSet('ex1.txt')
    ws = standRegres(xArr, yArr)
    #xArrSorted = np.mat(xArr).copy()
    #xArrSorted.sort(0)   #plot step by step
    yHat_lwlr = lwlrTest(xArr, xArr, yArr, 0.01)
    plot2DScatterLWLR(xArr, yArr, yHat_lwlr, ws, True)
    corr_coeaf_lwlr = np.corrcoef(yHat_lwlr.T, yArr)
    logger.debug("corr_coeaf of yHat_lwlr is %s", corr_coeaf_lwlr)
    return 


if __name__=='__main__':
    main()
    #test_standRegres()
    test_lwlr()
    removehandle()