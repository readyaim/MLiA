# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 03
tree.py
"""
import os
import sys
import re
import matplotlib.pyplot as plt
#import shutil
#import logging
import numpy as np
import operator
from math import log
from imp import reload

# Global variable definition here
debug =True
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
# Class definition here
class Class_definition(object):
    pass
# Function definition here

def calcShannonEnt(dataSet):
    """ Return the Shanno Entropy of class: -1 * sum(p(x) * log(p(x)))"""
    numEntries = len (dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] # The last column is the class
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1      #calc the num of labels
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/ numEntries  # probability of every label
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def creatDataSet():
    "Generate a dataSet"
    dataSet = [[1, 1, 'yes'],
                    [1, 1, 'yes'],
                    [1, 0 , 'no'],
                    [0, 1, 'no'],
                    [0, 1, 'no']]    
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
     
def splitDataSet(dataSet, axis, value):
    "pick up the dataSet that dataSet[axis]==value"
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet                    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   #the column of featues, discount labels
    baseEntropy = calcShannonEnt(dataSet)
    print("baseEntropy=%f"%baseEntropy)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #generate a list for feature[i]
        uniqueVals = set(featList)  #change list to set
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            prob = len(subDataSet)/float(len(dataSet))  #p(x)
            newEntropy += prob * calcShannonEnt(subDataSet) #sum(p(x)*entropy)
            #print("i=%d, value=%d, newEntropy=%f, shannonE=%f"\
            #    %(i, value, newEntropy, calcShannonEnt(subDataSet)))
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # why this is the best feature i? becuase this feature has made the entropy increase dramaticlly.
            bestInfoGain = infoGain #save the max infoGain
            bestFeature = i #save the feature index of max infoGain
        #print("infoGain=%f, bestInfoGain=%f, bestFeature=%f"\
          #          %(infoGain, bestInfoGain, bestFeature))
    return bestFeature  #return the index of best feature
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
    classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    "generate decision tree from training data set and labels"
    # class: yes(fish), no(for fish); 
    # feature: yes(for label1), no (for label2)
    # label: question Node. 
    classList = [example[-1] for example in dataSet]    # generate class list
    if classList.count(classList[0]) == len(classList):     # only 1 class remained
        return classList[0]                                           # return class, ignore the feature
    if len(dataSet[0]) == 1:        #only 1 labels remained, maybe with multiple classes
        return majorityCnt(classList)   # return majority class
    #else, with multi labels and multi classes
    bestFeat = chooseBestFeatureToSplit(dataSet)    #return the index of best feature
    bestFeatLabel = labels[bestFeat]    # get the best label 
    myTree = {bestFeatLabel:{}}     # add bestfeatlabel to tree
    del(labels[bestFeat])                   # del label
    featValues = [example[bestFeat] for example in dataSet]     #generate a features list for best feature.
    uniqueVals = set(featValues)    # generate a feature set of best features
    for value in uniqueVals:
        subLabels = labels[:]   #generate a copy of labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    storeTree(myTree,'classifierStorage', 'myTree')
    
    return myTree

def classify(inputTree, featLabels, testVec):
    #firstStr = inputTree.keys()[0]
    #"firstNode(feature)-> Vector->dict[vector]->labels"
    # TODO: what if featLabels
    firstStr = [k for k in inputTree.keys()][0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    #TODO: if firstStr not in featLabels
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel   

def storeTree(inputTree, filename, objname):
    import shelve
    with shelve.open(filename, writeback=True) as fw:
        fw[objname] = inputTree
    #with open(filename, 'wb') as fw:    
    #    pickle.dump(inputTree, fw)

def grabTree( treeName, filename='classifierStorage' ):
    #import pickle
    #with open(filename, 'rb') as fr:
    #    return pickle.load(fr)
    import shelve
    with shelve.open(filename) as fr:
        if treeName in fr.keys():
            return fr[treeName]
        else:
            print("%s not exist in classifierStorage"%treeName)
    return
# Test functions
def test_creatTree():
    dataSet, labels = creatDataSet()
    myTree = createTree(dataSet, labels)
    print(myTree)
def test_calcShannonEnt():
    dataSet, labels = creatDataSet()
    entropy = calcShannonEnt(dataSet)
    print(entropy)
    dataSet[0][2]='maybe'
    entropy = calcShannonEnt(dataSet)
    print(entropy)
    pass
def test_splitDataSet():
    dataSet, labels = creatDataSet()
    Dat = splitDataSet(dataSet, 0, 1)
    Dat2 = splitDataSet(dataSet, 0, 0)
    pass
def test_chooseBestFeatureToSplit():
    mydat, labels = creatDataSet()
    a = chooseBestFeatureToSplit(mydat)
    print(a)
    pass
def test_classify():
    mydat, labels = creatDataSet()
    import treePlotter
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    print(classify(myTree, labels, [1, 0]))
    print(classify(myTree, labels, [1, 1]))

def test_store_tree():
    mydat, labels = creatDataSet()
    import treePlotter
    myTree = treePlotter.retrieveTree(0)
    print(myTree)
    storeTree(myTree,'classifierStorage.txt')
    grabTree('classifierStorage.txt')

def test_lenses():
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)
        print(lensesTree)
        import treePlotter
        treePlotter.createPlot(lensesTree)

def test_classify_lense():
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    inputTree = grabTree('myTree')
    #print(inputTree)
    print(classify(inputTree, labels, ['young', 'hyper', 'no', 'reduced']))
    with open('lenses.txt') as fr:
        for line in fr:
            linelist = line.strip().split('\t')
            #print("the label shall be %s, the label predicted is %s"\
            #        %(linelist[-1], classify(inputTree, labels, linelist[:-1])))
            if linelist[-1] == classify(inputTree, labels, linelist[:-1]):
                print("correct")
            else:
                print("wrong")

def main():
    #test_chooseBestFeatureToSplit()
    #test_creatTree()
    #test_classify()
    #test_lenses()
    test_classify_lense()


if __name__=='__main__':
    #datingClassTest()
    main()
    