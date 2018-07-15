# -*- coding: utf-8 -*-   
"""
Machine Learning in Action
Chapter 04
bayes.py
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

# Class definition here
class Class_definition(object):
    pass
# Function definition here
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
    'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', \
    'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', \
    'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
    'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList, classVec
def createVocabList(dataSet):
    vocabSet = set([]) 
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) 
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: 
            print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #p0Num = np.zeros(numWords)
    #p1Num = np.zeros(numWords)
    #p0Denom = 0
    #p1Denom = 0 
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] 
            p1Denom += sum(trainMatrix[i]) 
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p0Vect = p0Num/p0Denom
    #p1Vect = p1Num/p1Denom
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1) 
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))        
    
def junk_email_dataset():
    mySent=r"This book is the best book on Python or M.L. I have ever laid eyes upon"
    mySent.split()
    import re
    regEx = re.compile("\\W")
    listOfTokens = regEx.split(mySent)
    print(listOfTokens)

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    #for i in range(1,26):
    for i in range(1,26):
        #print(i)
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))  #uniform distribuion (1/D)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]    

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]
    for i in range(20):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != \
            classList[docIndex]:
            errorCount += 1
    print ('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print (item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print ("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **")
    for item in sortedNY:
        print (item[0])
    
# Test functions
def test_createVocabList():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    print(setOfWords2Vec(myVocabList, listOPosts[0]))
    print(setOfWords2Vec(myVocabList, listOPosts[3]))
def test_trainNB0():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0v, p1v, pAb = trainNB0(trainMat, listClasses)
    print(p0v, p1v, pAb)

def test_rss():
    import feedparser
    ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    vocabList,pSF,pNY=localWords(ny,sf)
    getTopWords(ny,sf)

def main():
    pass
    #test_createVocabList()
    #test_trainNB0()
    #testingNB()
    #spamTest()
    #test_rss()
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


if __name__=='__main__':
    #datingClassTest()
    main()
    