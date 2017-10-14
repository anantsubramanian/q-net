import numpy as np
import tensorflow as tf
import os
import sys
import json
import random

from collections import Counter, defaultdict
from itertools import count
import pdb

sys.path.append("./src/")
from utils.tf_dl_utils import TfRNNCell, TfTraining
from proto import io
from proto import CoreNLP_pb2
from proto import dataset_pb2
from proto import training_dataset_pb2
from utils.squad_utils import ReconstructStrFromSpan, StandarizeToken

w2i = defaultdict(count(0).next)

class QaData(object):
    def __init__(self, title, qaId, query, context, 
        ans, ansToken, paraId, senId, pAnsId):
        '''
        @param title: indicate which article it belongs to.
        @param id: the unique id for qa pairs
        @param query, context, ans: list of int representation for RNN use
        @param ansToken: the tokens of the answer (for evaluation purpose)
        @param paraId: which paragraph the qa is from
        @param senId: which sentence the qa is from
        '''
        self.title = title
        self.id = qaId
        self.query = query
        self.context = context
        self.ans = ans
        self.ansToken = ansToken
        self.ansStr = ReconstructStrFromSpan(ansToken, (0, len(ansToken) ) )
        self.pAnsParaId = paraId
        self.pAnsSenId = senId
        self.pAnsId = pAnsId


class QaPrediction(object):
    def __init__(self, title, qaId, ansStr, paraId=None, 
        senId=None, queryStr=None, ansToken=None, score=0, contextToken=None):
        '''
        @param title: the title of the article to which the qa belong.
        @param ansStr: reconstructed string from tokens.
        @param paraId, senId: the location of the span in the article
        @param queryStr: optional, will be easy for debugging
        @param score: optional, the score of the answer for debuging 
        (e.g. within topK prediction)
        @param contextToken: the context sentence of the prediction
        '''
        self.title = title
        self.qaId = qaId
        self.ansStr = ansStr
        self.ansToken = ansToken
        self.paraId = paraId
        self.senId = senId
        self.queryStr = queryStr
        self.score = 0


class Agent(object):
    '''
    base class for qa agent
    '''
    def __init__(self, floatType, idType, lossType, articleLevel=True):
        '''
        loss type has to be either "max-margin" or "nce" 
        '''
        self.nVocab = None
        self.trainVar = None
        self.sampleIter = 0
        self.articleLevel = articleLevel
        # a list serving for debuging and run-time assertion
        self.debug = []


    def LoadCandidateData(self, dataFile):
        dataIn = io.ReadArticles(dataFile, cls=training_dataset_pb2.TrainingArticle)
        dataDict = dict()
        for data in dataIn:
            dataDict[data.title] = data
        return dataDict


    def LoadOrigData(self, dataFile):
        dataIn = io.ReadArticles(dataFile, cls=dataset_pb2.Article)
        dataDict = dict()
        for data in dataIn:
            dataDict[data.title] = data
        return dataDict


    def LoadTrainData(self, candFile, origFile, doDebug=False):
        self.trainCandData = self.LoadCandidateData(candFile)
        self.trainOrigData = self.LoadOrigData(origFile)
        if doDebug:
            title = self.trainCandData.keys()[0]
            self.trainCandData = {title : self.trainCandData[title] }
            self.trainOrigData = {title : self.trainOrigData[title] }


    def LoadEvalData(self, candFile, origFile, doDebug=False):
        self.evalCandData = self.LoadCandidateData(candFile)
        self.evalOrigData = self.LoadOrigData(origFile)
        if doDebug:
            title = self.evalCandData.keys()[0]
            self.evalCandData = {title : self.evalCandData[title] }
            self.evalOrigData = {title : self.evalOrigData[title] }


    def LoadVocab(self, vocabPath):
        print "Overriding LoadVocab"        
#with open(os.path.join(vocabPath, "word2id_train+dev.json"), "r") as fp:
#            self.wordToId = json.load(fp)
#        with open(os.path.join(vocabPath, "id2word_train+dev.json"), "r") as fp:
#            self.idToWord = json.load(fp)


    def TokenToId(self, tokens):
#return [self.wordToId[token.word.lower() ] for token in tokens]
        return [w2i[token.word.lower() ] for token in tokens]


    def IdToWord(self, ids):
#return [self.idToWord[idx] for idx in ids]
        i2w = {k:j for j,k in w2i.iteritems()} 
        return [i2w[idx] for idx in ids]


    def ShuffleData(self, seed=0):
        random.seed(seed)
        random.shuffle(self.trainSamples)


    def PrepareData(self, doTrain=True):
        '''
        use after the self.trainCandData and self.trainOrigData are loaded.
        The data samples are represented as RnnQaData in self.samples. 
        It may refer to self.trainCandidates (containing integer representation
        of candidate spans). It prepare the following for future use
        1. self.train/evalCandidates: span interger representation in article->paragraph->sentence hierarchy.
        2. self.train/evalCandGlobalId: map the span in the hierarchy to the global index in the article scope.
        3. self.train/evalSamples: sample information instances for training and evaluation
        They are the derived from self.train/evalCandData and self.train/evalOrigData
        Note we use self.train/evalCandGlobalId to get the global cand id given para and sentence idx
        We also use self.train/evalcandLocalId to get the para and sentence id given the global id
        '''
        def GetTokenFromSpan(candidateAns, sentence):
            '''
            @param candicateAns: a protobuf CandidateAnswer message.
            @param article: a protobuf Document message.
            Browse https://github.com/JianGoForIt/squad/blob/master/src/proto/dataset.proto
            and training_dataset.proto for more details
            '''
            pId = candidateAns.paragraphIndex
            sId = candidateAns.sentenceIndex
            tId = candidateAns.spanBeginIndex
            tLen = candidateAns.spanLength
            tokens = sentence.token
            tokens = tokens[tId:(tId + tLen) ]
            return tokens

        # candidates are index first by title
        if doTrain == True:
            self.trainCandidates = dict()
            self.trainCandGlobalId = dict()
            self.trainSamples = []
            titles = self.trainCandData.keys()
        else:
            self.evalCandidates = dict()
            self.evalCandGlobalId = dict()
            self.evalSamples = []
            titles = self.evalCandData.keys()
        for title in titles:
            if doTrain == True:
                tData = self.trainCandData[title]
                article = self.trainOrigData[title]
            else:
                tData = self.evalCandData[title]
                article = self.evalOrigData[title]
            assert tData.title == article.title
            assert tData.title == title
            # prepare integer representation of candidate spans
            candidates = [list() for i in range(len(article.paragraphs) ) ]
            candidateGlobalId = [list() for i in range(len(article.paragraphs) ) ]
            for iPara in range(len(article.paragraphs) ):
                paragraph = article.paragraphs[iPara]
                for iSen in range(len(paragraph.context.sentence) ):
                    candidates[iPara] = [list() for i in range(len(paragraph.context.sentence) ) ]
                    candidateGlobalId[iPara] = [list() for i in range(len(paragraph.context.sentence) ) ]

            for i, candAns in enumerate(tData.candidateAnswers):
                pId = candAns.paragraphIndex
                sId = candAns.sentenceIndex
                s = article.paragraphs[pId].context.sentence[sId]
                tokens = GetTokenFromSpan(candAns, s)
                tokens = StandarizeToken(tokens)
                if len(tokens) == 0:
                    continue
                intRep = self.TokenToId(tokens)
                candidates[pId][sId].append(intRep)
                candidateGlobalId[pId][sId].append(i)

                # TODO DEBUG there is no empty sentence
                assert len(candidates[pId][sId] ) != 0
                assert len(candidateGlobalId[pId][sId] ) != 0

            if doTrain == True:
                self.trainCandidates[title] = candidates
                self.trainCandGlobalId[title] = candidateGlobalId
            else:
                self.evalCandidates[title] = candidates
                self.evalCandGlobalId[title] = candidateGlobalId
            # prepare query interger representation
            queryDict = dict()
            ansDict = dict()
            ansTokenDict = dict()
            for paragraph in article.paragraphs:
                for qa in paragraph.qas:
                    qIntRep = self.TokenToId(qa.question.sentence[0].token)
                    # remove the last token if it is "." 
                    # also remove the first token if it is The or the
                    for answer in qa.answers:
                      if len(answer.sentence) == 0:
                          print qa.id, " has no answer."
                          continue
                      else:
                          tokens = answer.sentence[0].token
                          tokens = StandarizeToken(tokens)
                          if len(tokens) == 0:
                              continue
                          aIntRep = self.TokenToId(tokens)
                    queryDict[qa.id] = qIntRep
                    ansDict[qa.id] = aIntRep
                    ansTokenDict[qa.id] = tokens
                    
            # construct sample struct
            for qa in tData.questions: 
                if qa.id in ansDict.keys():
                    pAns = tData.candidateAnswers[qa.correctAnswerIndex]
                    paraId = pAns.paragraphIndex
                    senId = pAns.sentenceIndex
                    pAnsId = qa.correctAnswerIndex

                    query = queryDict[qa.id]
                    ans = ansDict[qa.id]
                    ansToken = ansTokenDict[qa.id]
                    contextToken = article.paragraphs[paraId].context.sentence[senId].token
                    context = self.TokenToId(contextToken)

                    sample = QaData(title=title, qaId=qa.id, query=query, context=context,
                        ans=ans, ansToken=ansToken, paraId=paraId, senId=senId, pAnsId=pAnsId)
                    if doTrain == True:
                        self.trainSamples.append(sample)
                    else:
                        self.evalSamples.append(sample)

        if doTrain == True:
            print "Prepared ", len(self.trainSamples), " training samples."
        else:
            print "Prepared ", len(self.evalSamples), " evaluation samples."
        self.idToWord = {j:k for k,j in w2i.iteritems()}
        self.wordToId = dict(w2i)


        # # self.evalSamples = self.trainSamples
        # # DEBUG
        # if doTrain:
        #     newSamples = []
        #     nCoverCnt = 0
        #     unkCnt = 0
        #     for sample in self.trainSamples:
        #         if len(newSamples) >= self.batchSize + 1:
        #             break
        #         title = sample.title
        #         paraId = sample.pAnsParaId
        #         senId = sample.pAnsSenId 
        #         if sample.ans in self.trainCandidates[title][paraId][senId]:
                    
        #     #         print "test preparation ", sample.id, paraId, senId
        #     #         print sample.ans#, self.trainCandidates[title][paraId][senId]

        #     #         # raw_input("done ")

        #             newSamples.append(sample)
        #     #         nCoverCnt += 1

        #     # # print "test sample quality", str(nCoverCnt / float(len(newSamples) ) )
        #     # # print "quality 2 ", str(unkCnt / float(len(newSamples) ) ) 

        #     # #         # if sample.id == "570d50a5fed7b91900d45e80":
        #     # #         #     newSamples.append(sample)
        #     # # # #     #     print ""
        #     # # #             print "sample info ", title , paraId, senId, sum( [len(candList) for candList in self.trainCandidates[title][paraId] ] )
        #     # # # #     #     print sample.ans, sum( [len(staff) for staff in self.trainCandidates[title][paraId] ] )
        #     # # # #     #     assert sample.pAnsId in self.trainCandGlobalId[title][paraId][senId]
        #     # # # #     #     print "all cand 0 ", [cand for candList in self.trainCandGlobalId[title][paraId] for cand in candList]


        #     self.trainSamples = newSamples[1:]
        #     self.evalSamples = newSamples[1:]
        # else:
        #     self.evalSamples = self.trainSamples
        # print "final ", len(self.trainSamples), len(self.evalSamples)


        # # # # # # assert len(self.trainSamples) == 1
        # # # # # # print newSamples[0].id


    def NegativeSampling(self, sample, negSampleSize):
        '''
        Only use this function for negative sampling in training
        '''
        title = sample.title
        pAnsParaId = sample.pAnsParaId
        pAnsSenId = sample.pAnsSenId
        pAnsId = sample.pAnsSenId
        candidates = self.trainCandidates[title][pAnsParaId][pAnsSenId]
        negSample = [cand for cand in candidates if cand != sample.ans]

        negSampleContext = [sample.context] * len(negSample)
        negSampleContextLen = [len(sample.context) ] * len(negSample)
        negSampleLen = [len(cand) for cand in negSample]
        randNegSample = []
        randNegSampleLen = []
        randNegSampleContext = []
        randNegSampleContextLen = []

        # give additional id
        # the first part of negative samples are from the same 
        # sentence as the correct answer
        negSampleParaId = [sample.pAnsParaId] * len(negSample)
        negSampleSenId = [sample.pAnsSenId] * len(negSample)
        candidateGlobalId = self.trainCandGlobalId[title][sample.pAnsParaId][sample.pAnsSenId]
        negSampleSpanId = []
        for spanId, cand in zip(candidateGlobalId, candidates):
            if cand != sample.ans:
                negSampleSpanId.append(spanId)

        # get maximum number of negsamples, in case of the required number
        # can not be reached and prevent trapping in the while loop
        # maxNSpan = sum( [len(spans) for spans in self.trainCandidates[title][pAnsParaId] ] )
        maxRandNegSpan = 0
        candLocalId = list()
        if self.articleLevel:
            for iPara, spanPara in enumerate(self.trainCandidates[title] ):
                for iSen, spanSen in enumerate(spanPara):
                    if pAnsSenId == iSen and pAnsParaId == iPara:
                        continue
                    maxRandNegSpan += len(spanSen)
                    candLocalId += [ (iPara, iSen, iSpan) \
                        for iSpan in range(len(self.trainCandGlobalId[title][iPara][iSen] ) ) ] 
        else:
            for iSen, spanSen in enumerate(self.trainCandidates[title][pAnsParaId] ):
                if pAnsSenId == iSen:
                    continue
                maxRandNegSpan += len(spanSen)
                candLocalId += [ (pAnsParaId, iSen, iSpan) \
                    for iSpan in range(len(self.trainCandGlobalId[title][pAnsParaId][iSen] ) ) ] 

        nRandNegSample = min(negSampleSize - len(negSample), maxRandNegSpan)
        if nRandNegSample != 0:
            candId = np.random.choice(np.array(range(len(candLocalId) ) ), nRandNegSample, replace=False)
            candLocalId = np.array(candLocalId)[candId, :]

            # TODO there is possiblity that there is not enough randnegcandidates.
            # but it is unlikely to happen in our dataset
            for iCand in range(nRandNegSample):
                paraId = candLocalId[iCand, 0]
                senId = candLocalId[iCand, 1]
                spanId = candLocalId[iCand, 2]
                cand = self.trainCandidates[title][paraId][senId][spanId]
                randNegSample.append(cand)
                randNegSampleLen.append(len(cand) )
                contextToken = self.trainOrigData[title].paragraphs[paraId].context.sentence[senId].token
                context = self.TokenToId(contextToken)
                randNegSampleContext.append(context)
                randNegSampleContextLen.append(len(context) )
                # for mapping neg samples back to global span Id
                negSampleParaId.append(paraId)
                negSampleSenId.append(senId)
                negSampleSpanId.append(self.trainCandGlobalId[title][paraId][senId][spanId] )

            negSample += randNegSample
            negSampleLen += randNegSampleLen
            negSampleContext += randNegSampleContext
            negSampleContextLen += randNegSampleContextLen

        assert len(negSample) == len(negSampleParaId)
        assert len(negSample) == len(negSampleSenId)
        assert len(negSample) == len(negSampleSpanId)

        return negSample, negSampleLen, len(negSample), negSampleContext, negSampleContextLen, \
            negSampleParaId, negSampleSenId, negSampleSpanId


    # def NegativeSampling(self, sample, negSampleSize):
    #     '''
    #     Only use this function for negative sampling in training
    #     '''
    #     title = sample.title
    #     pAnsParaId = sample.pAnsParaId
    #     pAnsSenId = sample.pAnsSenId
    #     pAnsId = sample.pAnsSenId
    #     candidates = self.trainCandidates[title][pAnsParaId][pAnsSenId]
    #     negSample = [cand for cand in candidates if cand != sample.ans and (not set(cand) <= set(sample.ans) ) ]

    #     negSampleContext = [sample.context] * len(negSample)
    #     negSampleContextLen = [len(sample.context) ] * len(negSample)
    #     negSampleLen = [len(cand) for cand in negSample]
    #     randNegSample = []
    #     randNegSampleLen = []
    #     randNegSampleContext = []
    #     randNegSampleContextLen = []

    #     # give additional id
    #     # the first part of negative samples are from the same 
    #     # sentence as the correct answer
    #     negSampleParaId = [sample.pAnsParaId] * len(negSample)
    #     negSampleSenId = [sample.pAnsSenId] * len(negSample)
    #     candidateGlobalId = self.trainCandGlobalId[title][sample.pAnsParaId][sample.pAnsSenId]
    #     negSampleSpanId = []
    #     for spanId, cand in zip(candidateGlobalId, candidates):
    #         if cand != sample.ans and (not set(cand) <= set(sample.ans) ):
    #             negSampleSpanId.append(spanId)

    #     # get maximum number of negsamples, in case of the required number
    #     # can not be reached and prevent trapping in the while loop
    #     # maxNSpan = sum( [len(spans) for spans in self.trainCandidates[title][pAnsParaId] ] )
    #     maxRandNegSpan = 0
    #     candLocalId = list()
    #     if self.articleLevel:
    #         for iPara, spanPara in enumerate(self.trainCandidates[title] ):
    #             for iSen, spanSen in enumerate(spanPara):
    #                 if pAnsSenId == iSen and pAnsParaId == iPara:
    #                     continue
    #                 maxRandNegSpan += len(spanSen)
    #                 candLocalId += [ (iPara, iSen, iSpan) \
    #                     for iSpan in range(len(self.trainCandGlobalId[title][iPara][iSen] ) ) ] 
    #     else:
    #         for iSen, spanSen in enumerate(self.trainCandidates[title][pAnsParaId] ):
    #             if pAnsSenId == iSen:
    #                 continue
    #             maxRandNegSpan += len(spanSen)
    #             candLocalId += [ (pAnsParaId, iSen, iSpan) \
    #                 for iSpan in range(len(self.trainCandGlobalId[title][pAnsParaId][iSen] ) ) ] 

    #     nRandNegSample = min(negSampleSize - len(negSample), maxRandNegSpan)
    #     if nRandNegSample != 0:
    #         candId = np.random.choice(np.array(range(len(candLocalId) ) ), nRandNegSample, replace=False)
    #         candLocalId = np.array(candLocalId)[candId, :]

    #         # TODO there is possiblity that there is not enough randnegcandidates.
    #         # but it is unlikely to happen in our dataset
    #         for iCand in range(nRandNegSample):
    #             if 1:
    #                 continue
    #             paraId = candLocalId[iCand, 0]
    #             senId = candLocalId[iCand, 1]
    #             spanId = candLocalId[iCand, 2]
    #             cand = self.trainCandidates[title][paraId][senId][spanId]
    #             randNegSample.append(cand)
    #             randNegSampleLen.append(len(cand) )
    #             contextToken = self.trainOrigData[title].paragraphs[paraId].context.sentence[senId].token
    #             context = self.TokenToId(contextToken)
    #             randNegSampleContext.append(context)
    #             randNegSampleContextLen.append(len(context) )
    #             # for mapping neg samples back to global span Id
    #             negSampleParaId.append(paraId)
    #             negSampleSenId.append(senId)
    #             negSampleSpanId.append(self.trainCandGlobalId[title][paraId][senId][spanId] )

    #         negSample += randNegSample
    #         negSampleLen += randNegSampleLen
    #         negSampleContext += randNegSampleContext
    #         negSampleContextLen += randNegSampleContextLen

    #     assert len(negSample) == len(negSampleParaId)
    #     assert len(negSample) == len(negSampleSenId)
    #     assert len(negSample) == len(negSampleSpanId)

    #     return negSample, negSampleLen, len(negSample), negSampleContext, negSampleContextLen, \
    #         negSampleParaId, negSampleSenId, negSampleSpanId




    def ConstructGraph(self, batchSize):
        pass


    def ConstructEvalGraph(self, batchSize):
        pass

    def GetNextTrainBatch(self):
        pass


    def Predict(self):
        pass





