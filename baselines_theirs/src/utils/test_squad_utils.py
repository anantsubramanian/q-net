import numpy as np
import matplotlib.pyplot as plt
import pylab
import re
import itertools
import json
import collections
from squad_utils import *
from data_processor import DataProcessor


def TestVocabMapping():
    dataFile = "./dataset/samples/qa-dump-1460090355004_new.json"
    wordToIdFile = "./wordToId.json"
    idToWordFile = "./idToWord.json"
    dataProvider = DataProcessor(dataFile)
    dataProvider.BuildVocab()
    dataProvider.SaveVocab(wordToIdFile, idToWordFile)

    dataProvider.LoadVocab(wordToIdFile, idToWordFile)
    dataProvider.TranslateWordToIdPerArticle()
    data = dataProvider.data
    for title in data.keys():
        article = data[title]
        sentencesInId = article["textInSentencesInId"]
        sentencesInWordsFromId = dataProvider.TranslateIdToWord(sentencesInId)
        sentencesInWords = SentenceToWord(article["textInSentences"] )
        for s0, s1 in zip(sentencesInWords, sentencesInWordsFromId):
            assert len(s0) == len(s1)
            for w0, w1 in zip(s0, s1):
                assert w0 == w1
    print "Vocab Mapping test passed!"


def TestExtNegativeSampling():
    dataFile = "./dataset/samples/qa-dump-1460090355004_new.json"
    dataProvider = DataProcessor(dataFile)
    nNegSample = 100
    dataProvider.NegSampleExt(nNegSample)

    for title in dataProvider.data.keys():
        article = dataProvider.data[title]
        for i in range(len(article["answers"] ) ):
            for negSample in article["negExtSamples"][i]:
                print " ".join(SentenceToWord( (article["answers"][i], ) )[0] ), negSample
                assert " ".join(SentenceToWord( (article["answers"][i], ) )[0] ) in negSample
    print "Extension negative sampling test passed!"


def TestRandNegativeSampling():
    dataFile = "./dataset/samples/qa-dump-1460090355004_new.json"
    dataProvider = DataProcessor(dataFile)

    # dataProvider.data = {"Imamah_(Shia_doctrine)" : \
    #     dataProvider.data["Imamah_(Shia_doctrine)"] }

    nNegSample = 100
    dataProvider.NegSampleRand(nNegSample)

    for title in dataProvider.data.keys():
        article = dataProvider.data[title]
        for i in range(len(article["answers"] ) ):
            for negSample in article["negRandSamples"][i]:
                # print " ".join(SentenceToWord( (article["answers"][i], ) )[0] ), negSample
                assert " ".join(SentenceToWord( (article["answers"][i], ) )[0] ) != negSample
    print "Random negative sampling test passed!"

def GetLongestCommonSubList(s1, s2):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


if __name__ == "__main__":
    # TestVocabMapping()
    # TestExtNegativeSampling()
    TestRandNegativeSampling()






        






