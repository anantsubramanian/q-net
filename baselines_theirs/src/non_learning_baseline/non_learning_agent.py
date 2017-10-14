import numpy as np
import multiprocessing as mp
import random
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("./src/")
from proto import io
from utils.squad_utils import ReconstructStrFromSpan
from utils import squad_utils
from utils.squad_utils import LoadProtoData, DumpJsonPrediction
from utils.squad_utils import MultipleProcess, ObjDict
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.evaluator import QaEvaluator

class NonLearningAgent(object):
    def __init__(self, randSeed):
        self.randSeed = randSeed 
        self.data = None
        self.stopWords = None


    def LoadData(self, dataFile):
        # load dataset
        self.data = LoadProtoData(dataFile)
        # load stopwords
        self.LoadStopWords()
        

    def LoadStopWords(self):
        with open("./src/utils/stop_word.txt", "r") as fp:
            self.stopWords = set( [word.lower() for word in fp.read().splitlines() ] )


    def GetContextBigram(self, article):
        '''
        article is an protobuf object for apecific article
        '''
        return squad_utils.GetContextBigram(article)


    def GetContextUnigram(self, article):
        '''
        article is an protobuf object for apecific article
        '''
        return squad_utils.GetContextUnigram(article)


    def GetBigramBySentence(self, tokens):
        '''
        tokens is a list of proto message object tokens
        '''
        return squad_utils.GetBigramBySentence(tokens)


    def GetContextConstituentSpan(self, article):
        return squad_utils.GetContextConstituentSpan(article)


    # def DumpPredToDict(self, predFile):
    #     random.seed(self.randSeed)
    #     predDict = dict()
    #     for title in self.data.keys():
    #         for prediction in self.predictions[title]:
    #             for pred in prediction:
    #                 if type(pred["answer"] ) == list:
    #                     randId = random.randint(0, len(pred["answer"] ) - 1)
    #                     predSingle = {"id": pred["id"], "answer": pred["answer"][randId], "token": pred["token"][randId] }
    #                     pred = predSingle
    #                 predDict[pred["id"] ] = pred["answer"]
    #     with open(predFile, "w") as fp:
    #         json.dump(predDict.copy(), fp)

    def DumpPrediction(self, fileName):
        with open(fileName, "w") as fp:
            json.dump(self.predictions, fp, default=ObjDict)
