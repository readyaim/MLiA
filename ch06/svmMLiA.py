# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 06
svmMLiA.py
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
from time import sleep
# Global variable definition here
debug =True

# Class definition here
class Class_definition(object):
    pass
class optStruct(object):
    def __init__(self,dataMatIn, classLabels, C, toler, kTup=('lin',1)):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        # error cache
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag, Error Cache
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


# Function definition here
def plotBestFit(supVector, wei=np.array([[1],[1],[1]]), datasetFile = "testSet.txt", showWeights=True):
    import matplotlib.pyplot as plt
    #weights = wei.getA()        #.getA(): get array from matrix
    dataMat,labelMat=loadDataSet(datasetFile)
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcordSV =[]; ycordSV = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    for j in range(len(supVector)):
        xcordSV.append(supVector[j][0])
        ycordSV.append(supVector[j][1])
    #weights = wei.getA()        #.getA(): get array from matrix


    weights = wei        #.getA(): get array from matrix
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
def calcEk(oS, k):
    "Calc and return Ek "
    #fXk = float(np.multiply(oS.alphas,oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b     #np.multiply(x1,x2)=broadcasting multiply
    #Ek = fXk - float(oS.labelMat[k])
    #next steps:
    fXk = float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
def calcWs(alphas,dataArr,classLabels):
    "Calc weights of SVM"
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(X)
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin':
        K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A                   #X-Y
            K[j] = deltaRow*deltaRow.T         #||X-Y||^2
        K = np.exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    #inner loop, heuristic ????
    #choose the 2nd alphas
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]   #.A means "matrix to array"
    #validEcacheList = np.nonzero(oS.eCache[:,0].A) return like:  (array([1, 5], dtype=int32), array([0, 0], dtype=int32))
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            # choose j for maximum step size to (Ei, i)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
            # end
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand, 2nd-choice heuristic
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print ("L==H"); return 0
        #eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        # for next steps:
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print ("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #add to eCache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print ("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])    #update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        #b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        #b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        # for next steps
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else: 
            oS.b = (b1 + b2)/2.0
        return 1
    else: return 0
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all values
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print ("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print ("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print ("iteration number: %d" % iter)
    return oS.b,oS.alphas
        
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j
def selectJrand2(i,m):
    while(True):
        j = int(np.random.uniform(0,m))
        if j!=i:
            return j
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)  #(100, 2)
    labelMat = np.mat(classLabels).transpose()  #(100,2)
    b = 0
    m,n = np.shape(dataMatrix)  #m=100, n=2
    alphas = np.mat(np.zeros((m,1)))      #alphas.shape=(100,1)
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):  #100 iterations
            #np.multiply(alphas:100,1,labelMat_100.1)=(1,100)
            #fXi is the predict value for label
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])   #if checks if an example violates KKT conditions
            #err is less than 0.1% and alphas<0.6
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: 
                    print ("L==H")
                    continue    #end this trial
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: 
                    print ("eta>=0")
                    continue    #end this trial
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta      #re-caculate alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue      #end this trial
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])    #update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T -\
                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
                    labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print ("iteration number: %d" % iter)
    return b,alphas

def gen_ws_by_smoP():
    dataArr,labelArr = loadDataSet('testSet.txt')
    b,alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    np.shape(alphas[alphas>0])
    """
    print(alphas[alphas>0])
    supVector=[]
    for i in range(100):
        if alphas[i]>0.0: 
            print(dataArr[i], labelArr[i])
            supVector.append(dataArr[i])
    print(supVector)
    plotBestFit(supVector)
    """
    ws = calcWs(alphas, dataArr, labelArr)
    #print(ws)
    return ws, b

def classify_by_smoP(dataArr):
    ws, b = gen_ws_by_smoP()
    datMat = np.mat(dataArr)
    #rst = np.datMat[0] * np.mat(ws)+b
    rst = datMat[0] * np.mat(ws)+b
    return 1 if rst[0]>=0 else -1

# Test functions
def test_loadDataSet():
    dataArr, labelArr = loadDataSet("testSet.txt")
    print(dataArr)
    print(labelArr)
def test_smoSimple():
    dataArr, labelArr = loadDataSet("testSet.txt")
    b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    np.shape(alphas[alphas>0])
    print(alphas[alphas>0])
    supVector=[]
    for i in range(100):
        if alphas[i]>0.0: 
            print(dataArr[i], labelArr[i])
            supVector.append(dataArr[i])
    ws = calcWs(alphas, dataArr, labelArr)
    weights = np.zeros((3,1))
    weights[0] = b
    weights[1] = ws[0]
    weights[2] = ws[1]
    print("ws is ",ws)
    print("b is ",b)
    plotBestFit(supVector, weights)
    #plotBestFit(supVector)
def test_Jrandom():
    import time
    startime = time.time()
    for i in range(100000):
        selectJrand(50, 100)
    print("selectJrand time is %f"%(time.time()-startime))
    startime = time.time()
    for i in range(100000):
        selectJrand2(50, 100)
    print("selectJrand2 time is %f"%(time.time()-startime))

def test_smoP(datasetFile = 'testSet.txt'):
    dataArr,labelArr = loadDataSet(datasetFile)
    b,alphas = smoP(dataArr, labelArr, 0.6, 0.0001, 40, ('lin', 1))
    np.shape(alphas[alphas>0])
    print(alphas[alphas>0])
    supVector=[]
    
    for i in range(100):
        if alphas[i]>0.0: 
            print(dataArr[i], labelArr[i])
            supVector.append(dataArr[i])
    print(supVector)
    ws = calcWs(alphas, dataArr, labelArr)
    weights = np.zeros((3,1))
    weights[0] = b
    #weights[1:] = ws
    weights[1] = ws[0]
    weights[2] = ws[1]
    print("ws is ",ws)
    print("b is ",b)
    plotBestFit(supVector, weights, datasetFile)
    
    #print("-"*10)
    #supVector = dataArr[alphas>0]    
def test_classify_by_smoP():
    data = np.array([4, -10])
    rlt = classify_by_smoP(data)
    print("result is %d"%rlt)

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat=np.mat(dataArr); labelMat = np.mat(labelArr).transpose()
    svInd=np.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd];
    print ("there are %d Support Vectors" % np.shape(sVs)[0])
    m,n = np.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print ("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = np.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * np.multiply(labelSV,alphas[svInd]) + b
        if np.sign(predict)!=np.sign(labelArr[i]):
            errorCount += 1
    print ("the test error rate is: %f" % (float(errorCount)/m))


    
def main():
    #test_loadDataSet()
    #test_loadDataSet()
    #test_smoSimple()
    #test_smoP('testSetRBF.txt')
    #test_smoP()
    testRbf()
    #test_classify_by_smoP()
    #plotBestFit()
    #test_Jrandom()

if __name__=='__main__':
    main()
    