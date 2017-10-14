import numpy as np
import json
import random
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("./src/")
from proto import io
from utils.squad_utils import LoadProtoData
from utils.squad_utils import MultipleProcess
from utils.squad_utils import ReconstructStrFromSpan
from utils.squad_utils import GetContextConstituentSpan
from utils.squad_utils import GetLongestCommonSubList


class QaEvaluator(object):
    '''
    Takes in QaData list and QaPrediction dict.
    produce the results
    '''
    def __init__(self, idToWord=None, wordToId=None, metrics=None):
        '''
        @param metric: it is a list of string specifying which metric to use
        @param idToWord, wordToId: the same as idToWord, wordToId of 
        qaAgent classes
        '''
        self.qaSamples = None
        self.qaPredictions = None
        self.metrics = metrics
        self.idToWord = idToWord
        self.wordToId = wordToId


    def EvaluatePrediction(self, samples, predictions):
        '''
        @param origData: for contextRnn model, it is either
        model.trainOrigData or model.evalOrigData
        '''
        results = dict()
        self.qaSamples = samples
        self.qaPredictions = predictions
        for metric in self.metrics:
            if "exact-match" in metric:
                topK = int(metric.split("top-")[-1] )
                results["exact-match"] = \
                    self.GetExactMatchRate(topK)
            elif "in-sentence-rate" in metric:
                topK = int(metric.split("top-")[-1] )
                results["in-sentence-rate"] = \
                    self.GetInSentenceRate(topK)
            elif metric == "<unk>-freq-in-pred":
                results["<unk>-freq-in-pred"] = \
                    self.GetUnkWordStatInPred()
            elif "overlap-match" in metric:
                rate = float(metric.split("match-rate-")[-1].split("-")[0] )
                topK = int(metric.split("top-")[-1] )
                results["overlap-match"] = \
                    self.GetOverlapCount(rate, topK)
            else:
                raise Exception(metric + " evaluation not implemented!")
        return results


    def GetOverlapCount(self, rate, topK):
        cntOverlap = 0
        cntSample = len(self.qaSamples)
        for sample in self.qaSamples:
            qaId = sample.id
            gtWord = [token.word for token in sample.ansToken]
            for i in range(min(topK, len(self.qaPredictions[qaId] ) ) ):
                if self.qaPredictions[qaId][i].paraId == sample.pAnsParaId \
                    and self.qaPredictions[qaId][i].senId == sample.pAnsSenId:
                    predWord = [token.word for token in self.qaPredictions[qaId][i].ansToken]
                    commonSubWords = GetLongestCommonSubList(gtWord, predWord)
                    if len(commonSubWords) / float(len(predWord) ) >= rate:
                        cntOverlap += 1
                        break
        print "Top " + str(topK) + " overlap answer ", str(cntOverlap), " / ", str(cntSample), \
            " : ", str(cntOverlap/float(cntSample) )
        return cntOverlap / float(cntSample)


    def GetExactMatchRate(self, topK=1):
        cntExactMatch = 0
        cntSample = len(self.qaSamples)
        for sample in self.qaSamples:
            qaId = sample.id
            gt = sample.ansStr
            for i in range(min(topK, len(self.qaPredictions[qaId] ) ) ):
                pred = self.qaPredictions[qaId][i].ansStr
                if pred.lower().strip() == gt.lower().strip():
                    cntExactMatch += 1
                    break
        print "Top " + str(topK) + " exact match ", str(cntExactMatch), " / ", str(cntSample), \
            " : ", str(cntExactMatch/float(cntSample) )
        return cntExactMatch/float(cntSample)


    def GetInSentenceRate(self, topK=1):
        cntInSentence = 0
        cntSample = len(self.qaSamples)
        for sample in self.qaSamples:
            qaId = sample.id
            for i in range(min(topK, len(self.qaPredictions[qaId] ) ) ):
                pred = self.qaPredictions[qaId][i]
                if sample.pAnsParaId == pred.paraId \
                    and sample.pAnsSenId == pred.senId:
                    cntInSentence += 1
                    break
        print "Top " + str(topK) + " In-sentence match ", str(cntInSentence), " / ", str(cntSample), \
            " : ", str(cntInSentence/float(cntSample) )
        return cntInSentence/float(cntSample)


    def GetUnkWordStatInPred(self):
        '''
        @param dict: vocabulary containing <unk> and <pad>
        '''
        wordToId = self.wordToId
        idToWord = self.idToWord
        cntUnkWrongAnsGt = 0
        cntUnkWrongAns = 0
        cntUnkCorrectAnsGt = 0
        cntUnkCorrectAns = 0
        cntExactMatch = 0
        for sample in self.qaSamples:
            qaId = sample.id
            gt = sample.ansStr
            pred = self.qaPredictions[qaId][0].ansStr
            words = [token.word.lower() for token in sample.ansToken]
            wordsInId = [wordToId[word] for word in words]
            wordsInDictGt = [idToWord[idx] for idx in wordsInId]

            words = [token.word.lower() for token in self.qaPredictions[qaId][0].ansToken]
            wordsInId = [wordToId[word] for word in words]
            wordsInDictPred = [idToWord[idx] for idx in wordsInId]

            if pred.lower().strip() == gt.lower().strip():
                cntExactMatch += 1
                if "<unk>" in wordsInDictPred:
                    cntUnkCorrectAns += 1
                if "<unk>" in wordsInDictGt:
                    cntUnkCorrectAnsGt += 1
            else:
                if "<unk>" in wordsInDictPred:
                    cntUnkWrongAns += 1
                if "<unk>" in wordsInDictGt:
                    cntUnkWrongAnsGt += 1

        print "<Unk> involved in ", str(cntUnkCorrectAns / float(cntExactMatch) ), \
            " ( " + str(cntUnkCorrectAns) + "/" + str(cntExactMatch) + " ) correct prediction"
        print "<Unk> involved in ", str(cntUnkWrongAns / float(len(self.qaSamples) - cntExactMatch) ), \
            " ( " + str(cntUnkWrongAns) + "/" + str(len(self.qaSamples) - cntExactMatch) + " ) wrong prediction"
        print "<Unk> involved in ", str(cntUnkCorrectAnsGt / float(cntExactMatch) ), \
            " ( " + str(cntUnkCorrectAnsGt) + "/" + str(cntExactMatch) + " ) in GT of correct prediction"
        print "<Unk> involved in ", str(cntUnkWrongAnsGt / float(len(self.qaSamples) - cntExactMatch) ), \
            " ( " + str(cntUnkWrongAnsGt) + "/" + str(len(self.qaSamples) - cntExactMatch) + " ) in GT of wrong prediction"


# if __name__ == "__main__":
#     dataFile = "/Users/Jian/Data/research/squad/dataset/proto/qa-annotated-full-1460521688980_new.proto"
#     # predFile = "/Users/Jian/Data/research/squad/output/pred-1460521688980_new-context_score-1-1-10.json"
#     predFile = "/Users/Jian/Data/research/squad/output/test-pred.json"
#     agent = Evaluator(verbose=2)
#     agent.LoadDataset(dataFile)
#     agent.EvalConstituentCoverage()
#     # agent.LoadPrediction(predFile)
#     # agent.CountMatch()

