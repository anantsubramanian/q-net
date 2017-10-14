import numpy as np
import matplotlib.pyplot as plt
import re
import itertools
import sys
import os
import json
import multiprocessing as mp
import threading
import time

sys.path.append("./src/")
from utils.squad_utils import LoadJsonData, DumpJsonPrediction
from utils.squad_utils import ParseJsonData
from utils.squad_utils import TextToSentence, SentenceToWord


class MCSMFSAgent(object):
    '''
    Max Cover Sentence + Most Frequent Span Method for Full Passage QA
    '''
    def __init__(self, dataPerArticle, verbose=1):
        '''
        @param dataPerArticle: a dictionary index by title of the article
        each entry corresponds to an article and contains the following field
        "textInSentences": text is segmented into a list of sentences 
        "queries": list of questions
        "answers": list of answers in the same order of "queries"
        @param maxCoverSentences: a dictionary index by title of the article.
        each element is a list with each elements corresponds to a query 
        in the corresponding article
        @param verbose: 0 for no specific output
        1 for statistics over the whole dataset
        2 for output of statistics for each specific article
        '''
        self.data = dataPerArticle
        self.verbose = verbose
        self.maxCoverSentences = dict()
        self.predictions = dict()
        self.nCores = mp.cpu_count()
        with open("./src/utils/stop_word.txt") as fp:
            self.stopWords = set(fp.read().splitlines() )


    def GetMaxCoverSentencePerArticle(self, title, returnDict):
        '''
        get the list of document sentences covering the most words in query.
        Note this aims at processing sentence and queries form a single article
        @param title: the title of the article
        '''
        if self.verbose >= 1:
            print "Searching Max Cover Sentence for " + title
        sentences = self.data[title]["textInSentences"]
        queries = self.data[title]["queries"]
        stopWords = self.stopWords
        sentencesInWords = SentenceToWord(sentences)
        queriesInWords = SentenceToWord(queries)
        nSentence = len(sentences)
        nQuery = len(queries)
        coverCnt = np.zeros( (nQuery, nSentence) )
        for iQ in range(nQuery):
            query = set(queriesInWords[iQ] )
            for iS in range(nSentence):
                sentence = set(sentencesInWords[iS] )
                # note we count duplicated intersection words as 
                overlap = sentence.intersection(query) - stopWords
                coverCnt[iQ, iS] = len(overlap)
        maxCoverSentences = list()
        for iQ in range(nQuery):
            maxCoverId = np.where(coverCnt[iQ] == np.amax(coverCnt[iQ] ) )[0]
            maxCoverSentences.append( [sentences[i] for i in maxCoverId] )
        returnDict[title] = maxCoverSentences


    def GetMaxCoverSentence(self):
        queue = mp.Queue()
        procs = []
        manager = mp.Manager()
        returnDict = manager.dict()
        for title in self.data.keys():
            p = mp.Process(target=MCSMFSAgent.GetMaxCoverSentencePerArticle,
                args=(self, title, returnDict) )
            procs.append(p)
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            proc.terminate()
        # # for DEBUG
        # for title in self.data.keys():
        #   self.GetMaxCoverSentencePerArticle(title, returnDict)
        self.maxCoverSentences = returnDict


    def PredictMCSPureSpanPerArticle(self, title, returnDict):
        '''
        @param maxCoverSentences: list of list of sentences, each list 
        of sentences corresponds to a query. Note the maxCoverSentences
        and queries follow the same order
        @param queries: a list of query strings
        @return prediction: list of list of pure spans with no intersection
        to the query. Note longer spans goes before shorter ones.
        '''
        if self.verbose >= 1:
            print "predicting span for " + title
        maxCoverSentences = self.maxCoverSentences[title]
        queries = self.data[title]["queries"]
        ids = self.data[title]["qaIds"]
        queriesInWords = SentenceToWord(queries)
        prediction = list()
        assert len(maxCoverSentences) == len(ids)
        for id in range(len(maxCoverSentences) ):
            maxCover = maxCoverSentences[id]
            query = queriesInWords[id]
            maxCoverInWords = SentenceToWord(maxCover)
            prediction.append(list() )
            for sentence in maxCoverInWords:
                maxLen = np.zeros( (len(sentence) + 1, ) )
                # dynamic programming for longest span
                # avoid sub span of the longest one
                for i in range(len(sentence) ):
                    if sentence[i] not in set(query) - self.stopWords:
                        maxLen[i + 1] = maxLen[i] + 1
                for i in range(len(sentence) ):
                    if maxLen[i + 1] != 0 \
                        and (i + 1 == len(sentence) \
                        or maxLen[i + 1] >= maxLen[i + 2] ):
                        prediction[-1].append(sentence[int(i - maxLen[i + 1] + 1):(i + 1) ] )
                prediction[-1].sort(key=lambda x: len(x), reverse=True)
            for i in range(len(prediction[-1] ) ):
                prediction[-1][i] = " ".join(prediction[-1][i] )
            prediction[-1] = {"prediction": prediction[-1], "id": ids[id] }
        returnDict[title] = prediction


    def PredictMCSPureSpan(self):
        '''
        get the longest span without intersection with query strings.
        Intuitively, this should be a relatively high recall model.
        '''
        procs = []
        manager = mp.Manager()
        returnDict = manager.dict()
        for title in self.data.keys():
            p = mp.Process(target=MCSMFSAgent.PredictMCSPureSpanPerArticle,
                args=(self, title, returnDict) )
            procs.append(p)
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            proc.terminate()
        # # for DEBUG
        # for title in self.data.keys():
        #   self.PredictMCSPureSpanPerArticle(title, returnDict)
        self.predictions = returnDict


    # def EvalCoverageInSentenceList(self):
    #     '''
    #     evaluate how many spans are covered in the list of selected 
    #     max cover sentences 
    #     '''
    #     dataPerArticle = self.data
    #     maxCoverSentences = self.maxCoverSentences
    #     # count how many answers are successfully covered
    #     ansCntTotal = 0
    #     coveredAnsCntTotal = 0
    #     for title in dataPerArticle.keys():
    #         ansCnt = 0
    #         coveredAnsCnt = 0
    #         queries = dataPerArticle[title]["queries"]
    #         answers = dataPerArticle[title]["answers"]
    #         maxCovers = maxCoverSentences[title]
    #         assert len(queries) == len(answers)
    #         assert len(queries) == len(maxCovers)
    #         for i in range(len(queries) ):
    #             ansCnt += 1
    #             maxCoverJoin = " ".join(maxCovers[i] )
    #             if answers[i] in maxCoverJoin:
    #                 coveredAnsCnt += 1
    #         ansCntTotal += ansCnt
    #         coveredAnsCntTotal += coveredAnsCnt
    #         if self.verbose >= 2:
    #             print str(coveredAnsCnt) + " / " + str(ansCnt) + " covered by Max Cover Sentences for " + title
    #     if self.verbose >= 1:
    #         print str(coveredAnsCntTotal) + " / " + str(ansCntTotal) + " covered by Max Cover Sentences in total"


    # def EvalPureStringCoverage(self, topK=1):
    #     '''
    #     evaluate how many answers are successfully covered in the 
    #     longest pure string from max cover sentences.
    #     We will also check the how redudent is the longest pure span
    #     for successfully covered answer. it measure the ratio
    #     between length of answer and selected span
    #     ''' 
    #     dataPerArticle = self.data
    #     ansCntTotal = 0
    #     coveredAnsTotal = 0
    #     purityTotal = 0
    #     exactAnsTotal = 0
    #     for title in dataPerArticle.keys():
    #         article = dataPerArticle[title]
    #         queries = article["queries"]
    #         answers = article["answers"]
    #         answers = SentenceToWord(answers)
    #         predictions = self.predictions[title]
    #         assert len(predictions) == len(answers)
    #         coveredAns = 0
    #         exactAns = 0
    #         purity = 0
    #         for i in range(len(queries) ):
    #             ans = " ".join(answers[i] ).strip() 
    #             midPos = len(predictions[i]["prediction"] ) / 2
    #             nBefore = (topK - 1) / 2
    #             nAfter = topK - 1 - nBefore 
    #             beginPos = max(0, midPos - nBefore)
    #             endPos = min(midPos + nAfter, len(predictions[i]["prediction"] ) )
    #             predMedium = predictions[i]["prediction"][beginPos:(endPos + 1) ]
    #             assert len(predMedium) == topK or len(predMedium) == len(predictions[i]["prediction"] )
    #             for j in range(min(topK, len(predictions[i]["prediction"] ) ) ):
    #                 # pred = predictions[i]["prediction"][j]
    #                 pred = predMedium[j]
    #                 if ans in pred:
    #                     coveredAns += 1
    #                     purity += len(ans) / float(len(pred) )
    #                     break
    #             for j in range(min(topK, len(predictions[i]["prediction"] ) ) ):
    #                 pred = predMedium[j]
    #                 if ans == pred:
    #                     exactAns += 1
    #             # for j in range(min(topK, len(predictions[i]["prediction"] ) ) ):
    #             #     pred = predictions[i]["prediction"][j]
    #             #     if ans in pred:
    #             #         coveredAns += 1
    #             #         purity += len(ans) / float(len(pred) )
    #             #         break
    #             # for j in range(min(topK, len(predictions[i]["prediction"] ) ) ):
    #             #     pred = predictions[i]["prediction"][j]
    #             #     if ans == pred:
    #             #         exactAns += 1
    #         coveredAnsTotal += coveredAns
    #         ansCntTotal += len(answers)
    #         purityTotal += purity
    #         exactAnsTotal += exactAns
    #         if self.verbose >= 2:
    #             if coveredAns != 0:
    #                 print "Purity of longest pure span is " + str(purity / float(coveredAns) ) + " for " + title
    #             print str(coveredAns) + " / " + str(len(answers) ) + " (" + str(exactAns) + " exact) covered by top " + str(topK) + " pure span for " + title
    #     if self.verbose >= 1:
    #         print "Purity of longest pure span is " + str(purityTotal / float(coveredAnsTotal) ) + " in total"
    #         print str(coveredAnsTotal) + " / " + str(ansCntTotal) + " (" + str(exactAnsTotal) + " exact) covered by top " + str(topK) + " pure span in total"

       

if __name__ == "__main__":
    if not os.path.exists("./output"):
        os.mkdir("./output")
    inputFilePath = "./dataset/samples/qa-dump-1459575857305_new.json"
    outputFilePath = "./output/nlb-ans-dump-1459575857305_new.json"

    topK = 1
    assert topK > 0

    data = LoadJsonData(inputFilePath)
    dataPerArticle = ParseJsonData(data)

    agent = MCSMFSAgent(dataPerArticle, verbose=2)
    agent.GetMaxCoverSentence()
    agent.EvalCoverageInSentenceList()
    agent.PredictMCSPureSpan()
    agent.EvalPureStringCoverage(topK)

    DumpJsonPrediction(outputFilePath, agent.predictions)













