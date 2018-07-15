# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 03
treeplotter.py
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



def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",\
                                        xytext=centerPt, textcoords="axes fraction", \
                                        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
def createPlot_naive():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("a decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("a leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    #firstStr = myTree.keys()[0]
    firstStr = [k for k in myTree.keys()][0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #if type(secondDict[key]).__name__ =="dict":
        if isinstance(secondDict[key], dict):
            numLeafs +=getNumLeafs(secondDict[key]) #recursively calling
        else:
            numLeafs +=1
    return numLeafs            

def getTreeDepth(myTree):
    maxDepth = 0
    #firstStr = myTree.keys()[0]
    firstStr = [k for k in myTree.keys()][0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])   #recursively calling
        else:
            thisDepth = 1
        # to compare two/N branches of same level
        if thisDepth > maxDepth: 
            maxDepth = thisDepth
    return maxDepth
    
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                            {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                                {'no surfacing': {0: {'flippers': {0: {'head': {0: {'feather':{0:'yes', 1:'no'}}, 1: 'yes'}}, 1: {'feather':{0:'yes', 1:'no'}}}},1: 'no'}}]
                                    
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    "Plot text between child and parrent"
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    "plot Node from top to bottom with 1/Depth, plot leaf from left to right with 1/Width gap"
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)
    firstStr = [k for k in myTree.keys()][0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #first node
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD #y = y - 1/D, go to lower layer
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            #node
            plotTree(secondDict[key],cntrPt,str(key))       #recursively calling
        else:
            #leaf
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW #x = x + 1/N
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #y = y+1/D, go back to upper layer
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW    #in worse case, all leaves fall at half side(0.5)
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), 'tree')
    plt.show()    
# Test functions
def test_createPlot_naive(  ):
    createPlot_naive()
    return 0
    
def test_treePlotter(  ):
    myTree = retrieveTree(0)
    print("Tree(%d), NumLeafs = %d"%(0, getNumLeafs( retrieveTree(0))))
    print("Tree(%d), TreeDepth = %d"%(0, getTreeDepth( retrieveTree(0))))
    print("Tree(%d), NumLeafs = %d"%(1, getNumLeafs( retrieveTree(1))))
    print("Tree(%d), TreeDepth = %d"%(1, getTreeDepth( retrieveTree(1))))
    return    0
def test_createPlot():
    myTree = retrieveTree(2)
    createPlot(myTree)
def main():
    #test_createPlot_naive()
    #test_treePlotter()
    test_createPlot()
    


if __name__=='__main__':
    #datingClassTest()
    main()
    