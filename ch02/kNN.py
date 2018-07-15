# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 02
kNN.py
"""
import os
import sys
import re
import matplotlib.pyplot as plt
#import shutil
#import logging
import numpy as np
import operator
# Global variable definition here
debug =True

# Class definition here
class Class_definition(object):
    pass
# Function definition here

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] 
    # distance caclculation  
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet     #np.tile to extend the array
    sqDiffMat = diffMat**2
    sdDistances = sqDiffMat.sum(axis=1)
    distances = sdDistances**0.5
    # end 
    sortedDisIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        # Voting with lowest K distance
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort dictionary
    sortedClassCount = sorted (classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
def file2matrix( filename ):
    with open(filename, "r") as fr:
        lines = fr.readlines()
        num_lines = len(lines)
        ret_Mat = np.zeros((num_lines, 3))
        classLabelVector = []
        fr.seek(0)
    #with open(filename, "r") as fr:
        index = 0
        labeltype = 0
        classlabelDict={}
        for line in lines:
            listFromLine = line.strip().split("\t")
            ret_Mat[index, :] = listFromLine[0:3]
            if listFromLine[-1] not in classlabelDict.keys():
                labeltype +=1
                classlabelDict[listFromLine[-1]] =  labeltype
            
            classLabelVector.append(int(classlabelDict[listFromLine[-1]]))
            index += 1
        for key in classlabelDict.keys():
            print(key, classlabelDict[key])    
    return ret_Mat, classLabelVector

def autoNorm( dataSet ):
    """Normial Dating data, (m, 4)
    data = data/(max - min)
    """
    minVals = dataSet.min(0)    #array's min in axis=0. (1, 3)
    maxVals = dataSet.max(0)    #(1, 3)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)   #(1000,3)
    m = dataSet.shape[0]    #m=1000
    normDataSet = dataSet - np.tile(minVals, (m,1)) 
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int (lineStr[j])
        return returnVect        
def handwritingClassTest(  ):
    hwLabels = []
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/%s"%fileNameStr)
    testFileList = os.listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s"%fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
                %(classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
        print("\nthe total number of errors is: %d"%errorCount)
        print("\nthe total error rate is: %f"%(errorCount/float(mTest)))
    return 0
# Test functions
def sortTest():
    group,labels = createDataSet()
    sortedValue = classify0([0,0], group, labels, 3)
    print(sortedValue)

def datingDataPlot():    
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    # Generate largedose category
    largedoseLabel = np.array(datingLabels)
    largedose_index = largedoseLabel>1
    largedoseLabel[largedose_index] = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(datingDataMat[:,1], datingDataMat[:, 2], 15*np.array(datingLabels), 15*np.array(datingLabels))
    # Show largedose category
    ax2.scatter(datingDataMat[:,1], datingDataMat[:, 2], 15*largedoseLabel, 15*largedoseLabel)
    plt.show()
    normMat, ranges, minVals = autoNorm(datingDataMat)
pass    
        
def main():
    pass


def datingClassTest(  ):
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3) # code to be executed
        print("the classfier came back with: %d, the real answer is %d"%(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is %f"%(errorCount/float(numTestVecs))) 
    return 0   
# plot the one he like 
if __name__=='__main__':
    #datingClassTest()
    handwritingClassTest()
    