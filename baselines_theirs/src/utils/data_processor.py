import numpy as np
# import matplotlib.pyplot as plt
# import pylab
import re
import itertools
import json
import collections
import multiprocessing as mp
import random
import sys, os

sys.path.append("./src/")
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.squad_utils import *


def multipleProcess(agent, titleList, targetFunc, conservative=True, debug=False):
    '''
    target function is the one we want to execute
    for each article. When conservative == True, the num of threads
    is equal to the number of cores on the machine
    '''
    procs = []
    manager = mp.Manager()
    returnDict = manager.dict()
    if debug:
        for title in titleList:
          targetFunc(agent, title, returnDict)
    else:
        for title in titleList:
            p = mp.Process(target=targetFunc,
                args=(agent, title, returnDict) )
            procs.append(p)
        processor = MultiProcessorCPU(procs)
        processor.run(conservative)  
    return returnDict


class DataProcessor(object):
    '''
    the pre/postprocessor class for manipulating information from raw data
    For vocab generation: BuildVocab then translate between word and Id
    For negative sampling: generate extension/random negative samples
    separately. We can mix them when needed. 
    Using conservativeParallel prevents from spawning too much processes.
    '''
    def __init__(self, conservativeParallel=True, randSeed=0):
        # self.data = ParseJsonData(LoadJsonData(fileName) )
        self.wordToId = None
        self.randSeed = randSeed
        self.nCores = mp.cpu_count()
        self.conservativeParallel = conservativeParallel


    def LoadJsonData(self, fileName):
        self.data = ParseJsonData(LoadJsonData(fileName) )


    def LoadProtoData(self, fileName):
        self.data = LoadProtoData(fileName)


    def BuildVocabFromJson(self):
        '''
        read from json data file and construct a vocab repo from it
        Note self. idToWord is simply an array while WordToId is a dict.
        '''
        sentences = []
        for title in self.data.keys():
            article = self.data[title]
            sentences += article["textInSentences"]
        words = SentenceToWord(sentences)
        words = [w for wList in words for w in wList]
        counter = collections.Counter(words)
        countPairs = sorted(counter.items(), key=lambda x:(-x[1], x[0] ) )
        words, freq = list(zip(*countPairs) )
        self.wordToId = dict(zip(words, range(len(words) ) ) )
        self.idToWord = words
        print "constructed a dict with " + str(len(words) ) + " entries."
        return self.wordToId, self.idToWord


    def BuildVocabFromProto(self):
        '''
        Build vocabulary dictionary after loading from protobuf data
        via self.LoadProtoData.
        '''
        contextWords = []
        for title in self.data.keys():
            print "processing context in ", title
            article = self.data[title]
            for paragraph in article.paragraphs:
                for s in paragraph.context.sentence:
                    contextWords += [token.word.lower() for token in s.token]
        contextWords = set(contextWords)

        queryWords = []
        ansWords = []
        for title in self.data.keys():
            print "processing qa in ", title
            article = self.data[title]
            for paragraph in article.paragraphs:
                for qa in paragraph.qas:
                    if len(qa.answer.sentence) == 0:
                        print qa.id
                        continue
                    qS = qa.question.sentence[0]
                    aS = qa.answer.sentence[0]
                    queryWords += [token.word.lower() for token in qS.token]
                    ansWords += [token.word.lower() for token in aS.token]
        queryWords = set(queryWords)
        ansWords = set(ansWords)

        words = set( [] )
        words = words.union(contextWords).union(queryWords).union(ansWords)
        words = list(words)
        self.wordToId = dict(zip(words, range(len(words) ) ) )
        self.idToWord = words
        print "n words in context query and ans: ", len(contextWords), len(queryWords), len(ansWords)
        print "constructed a dict with " + str(len(words) ) + " entries."
        return self.wordToId, self.idToWord


    def SaveVocab(self, wordToIdFile, idToWordFile):
        with open(wordToIdFile, "w") as outFile:
            json.dump(self.wordToId, outFile)
        with open(idToWordFile, "w") as outFile:
            json.dump(self.idToWord, outFile)


    def LoadVocab(self, wordToIdFile, idToWordFile):
        with open(wordToIdFile, "r") as inFile:
            self.wordToId = json.load(inFile)
        with open(idToWordFile, "r") as inFile:
            self.idToWord = json.load(inFile)


    def TranslateWordToId(self, sentences):
        '''
        sentences is a list of sentence(full string)
        '''
        wordToId = self.wordToId
        sentenceInWords = SentenceToWord(sentences )
        sentencesInId = \
            [ [wordToId[w] for w in s] for s in sentenceInWords]
        return sentencesInId


    def TranslateWordToIdPerArticle(self):
        for title in self.data.keys():
            article = self.data[title]
            self.data[title]["textInSentencesInId"] = \
                self.TranslateWordToId(article["textInSentences"] )
             

    def TranslateIdToWord(self, sentencesInId):
        '''
        sentencesInId is a list of int list. Each int list
        corresponds to a sentence.
        '''
        idToWord = self.idToWord
        sentencesInWords = \
            [ [idToWord[i] for i in s] for s in sentencesInId]
        return sentencesInWords


    def NegSampleExtPerArticle(self, title, returnDict):
        '''
        Do negative sampling by extending one word to left or right
        each time. Note we traverse all the sentence covering the 
        correct answer. 
        '''
        print "Generating extension negative samples for " + title
        answers = self.data[title]["answers"]
        contextInSentences = self.data[title]["textInSentences"]
        context = SentenceToWord(contextInSentences)
        negAnswers = list()
        answersInWords = SentenceToWord(answers)
        for i in range(len(answers) ):
            # remove the effect of additional , . ... e.g october 23, 2001
            ans = " ".join(answersInWords[i] ).strip()
            negAns = []
            for j in range(len(context) ):
                if answers[i] not in contextInSentences[j]:
                    continue
                negAns += [context[j][p:q] for p in range(len(context[j] ) ) \
                    for q in range(p + 1, len(context[j] ) + 1 ) \
                    if (ans in " ".join(context[j][p:q] ).strip() \
                    and ans != " ".join(context[j][p:q] ).strip() ) ]
            negAns = sorted(negAns, key=lambda x: len(x) )
            negAns = [" ".join(a).strip() for a in negAns]
            negAnsUniq = []
            [negAnsUniq.append(a) for a in negAns if a not in negAnsUniq]
            # assert len(negAnsUniq) != 0 # full sentence answer may cause this.
            negAnswers.append(negAnsUniq[0:min(self.nNegSample, len(negAnsUniq) ) ] )
        returnDict[title] = negAnswers


    def NegSampleExt(self, nSample):
        self.nNegSample = nSample
        negSamples = multipleProcess(self, self.data.keys(),
            DataProcessor.NegSampleExtPerArticle, 
            conservative=self.conservativeParallel)
        for title in self.data.keys():
            assert len(negSamples[title] ) == len(self.data[title]["qaIds"] )
            self.data[title]["negExtSamples"] = negSamples[title]   


    def NegSampleRandPerArticle(self, title, returnDict):
        '''
        we use random negative sampling to make the num of 
        negative sampling reach the amount specified by 
        self.nNegSample.
        '''
        print "Generating random negative samples for " + title
        answers = self.data[title]["answers"]
        contextInSentences = self.data[title]["textInSentences"]
        context = SentenceToWord(contextInSentences)

        negAnswers = list()
        answers = SentenceToWord(answers)
        startPos = [0,] + [len(context[i] ) for i in range(len(context) ) ]
        startPos = np.cumsum(np.array(startPos) )
        nContextSentences = len(context)
        context = [w for s in context for w in s]

        for i in xrange(len(answers) ):
            ansLen = len(answers[i] )
            ans = " ".join(answers[i] ).strip()
            # we need to ensure the entity is not crossing sentences
            pos = [p for j in range(nContextSentences) \
                for p in range(startPos[j], startPos[j + 1] - ansLen + 1) ]
            posRand = np.random.choice(np.array(pos), 
                min(self.nNegSample, len(pos) ), replace=False)
            negSamples = [" ".join(context[i:(i + ansLen) ] ).strip() for i in posRand \
                if ans != " ".join(context[i:(i + ansLen) ] ).strip() ]
            # note if the right answer is a full sentence longer than all the others, this may break
            assert len(negSamples) != 0 
            negAnswers.append(negSamples)
        returnDict[title] = negAnswers


    def NegSampleRand(self, nSample):
        '''
        randomly generate negative sample answer with the 
        same length as answer
        '''   
        self.nNegSample = nSample
        negSamples = multipleProcess(self, self.data.keys(),
            DataProcessor.NegSampleRandPerArticle, 
            conservative=self.conservativeParallel)
        for title in self.data.keys():
            assert len(negSamples[title] ) == len(self.data[title]["qaIds"] )
            if "negRandSamples" not in self.data[title].keys():
                self.data[title]["negRandSamples"] = list()
            self.data[title]["negRandSamples"] += negSamples[title]


    def MergeVocab(self, vocabPath):
        with open(os.path.join(vocabPath, "word2id_train.json") ) as fp:
            self.wordToId = json.load(fp)
        for key in self.wordToId.keys():
            self.wordToId[key] += 2
        self.wordToId["<pad>"] = 0
        self.wordToId["<unk>"] = 1
        with open(os.path.join(vocabPath, "id2word_train.json") ) as fp:
            self.idToWord = json.load(fp)
        self.idToWord = ["<pad>", "<unk>"] + self.idToWord
        self.nVocab = len(self.idToWord)
        # TODO change this ugly patch on unknow works from dev set.
        # and recover the assert
        with open(os.path.join(vocabPath, "id2word_dev.json") ) as fp:
            additionalWord = json.load(fp)
            for word in additionalWord:
                if word not in self.wordToId.keys():
                    self.wordToId[word] = self.wordToId["<unk>"]
                    self.idToWord.append("<unk>") 
        with open(os.path.join(vocabPath, "word2id_train+dev.json"), "w") as fp:
            json.dump(self.wordToId, fp) 
        with open(os.path.join(vocabPath, "id2word_train+dev.json"), "w") as fp:
            json.dump(self.idToWord, fp)


    def GetInitWordVec(self, vocabPath, nEmbedDim, fileName, outFile, stdDev=0.1):
        self.MergeVocab(vocabPath)
        unkId = self.wordToId["<unk>"]
        padId = self.wordToId["<pad>"]
        cntPreTrainVec = 0
        cntTotal = 0
        initEmbedding = np.random.randn(self.nVocab, nEmbedDim) * stdDev
        with open(fileName, "r") as fp:
            keySet = set(self.wordToId.keys() )
            for line in fp:
                word = line.split(" ")[0]
                cntTotal += 1
                # print "processing ", word
                if unicode(word, "utf-8") in keySet:
                    wordId = self.wordToId[unicode(word, "utf-8") ]
                    if wordId == unkId or wordId == padId:
                        continue
                    vecStr = line.split(" ")[1:]
                    # if word == "love":
                    #     print " love ", [float(s) for s in vecStr]
                    initEmbedding[wordId, :] = [float(s) for s in vecStr]
                    cntPreTrainVec += 1
                if cntTotal % 10000 == 0:
                    print "done ", cntTotal
        with open(outFile, "w") as fp:
            np.save(outFile, initEmbedding)

        # assert for one word
        # theId = self.wordToId["love"]
        # print "for the ", initEmbedding[theId, :]
        print "get pretrainWord done for ", cntPreTrainVec, np.linalg.norm(initEmbedding)


if __name__ == "__main__":
    # fileName = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    # # fileName = "./qa-annotated-train-1460521688980_new.proto"
    dataAgent = DataProcessor()
    # dataAgent.LoadProtoData(fileName)
    # dataAgent.BuildVocabFromProto()
    # dataAgent.SaveVocab(wordToIdFile="./dataset/proto/vocab_dict/word2id_dev.json", idToWordFile="./dataset/proto/vocab_dict/id2word_dev.json")

    # vocabPath = "./dataset/proto/vocab_dict"
    # dataAgent.MergeVocab(vocabPath)

    vocabPath = "./dataset/proto/vocab_dict"
    dataAgent.GetInitWordVec(vocabPath, 100, "./dataset/glove/glove.6B.100d.txt", "./dataset/glove/glove.6B.100d.init")
