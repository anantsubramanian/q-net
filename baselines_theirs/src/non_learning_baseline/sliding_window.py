import pdb
import numpy as np
import multiprocessing as mp
import random
import json
from scipy.spatial import distance
import tensorflow as tf
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("./src/")
from proto import io
from utils.squad_utils import ReconstructStrFromSpan
from utils import squad_utils
from utils.squad_utils import LoadProtoData, DumpJsonPrediction
from utils.squad_utils import MultipleProcess
from utils.multiprocessor_cpu import MultiProcessorCPU
from utils.evaluator import QaEvaluator
from non_learning_baseline.non_learning_agent import NonLearningAgent
from learning_baseline.agent import QaPrediction, Agent

class SlidingWindowAgent(NonLearningAgent):
    def __init__(self, lambDist=1, lenPenalty=0, randSeed=0, articleLevel=True, topK=1):
        NonLearningAgent.__init__(self, randSeed)
        self.lambDist = lambDist
        self.randSeed = randSeed
        self.lenPenalty = lenPenalty
        self.articleLevel = articleLevel
        self.topK = topK


    def GetStopWords(self, stopWords):
        self.stopWords = stopWords


    def GetSlidingDistScore(self, paragraph, question, option):
        return self.GetSlidingWindowScore(paragraph, question, option)\
            - self.GetMinDistance(paragraph, question, option) * self.lambDist


    def GetSlidingWindowScore(self, paragraph, question, option):
        '''
        Get the sliding window score. 
        @param paragraph, questions, option: a list of token words
        '''
        def ic(w, p):
            return np.log(1 + (1.0 / p.count(w)))
        o = [ele for ele in option if ele not in self.stopWords]
        p = [ele for ele in paragraph if ele not in self.stopWords]
        q = [ele for ele in question if ele not in self.stopWords]
        s = set(o + q)
        scores = []
        if len(p) == 0 or len(s) == 0:
            return sys.float_info.min
        else:
            for i in range(len(p)):
                sum = 0
                for j in range(len(s)):
                    if i + j < len(p) and p[i + j] in s:
                        sum = sum + ic(p[i + j], p)
                scores.append(sum)
            return np.max(scores) - self.lenPenalty * len(option)


    def GetMinDistance(self, paragraph, question, option):
        """Distance based add to sliding."""
        def distance_between(q, o, p):
            """Minimum distance.

            Minimum number of words between an
            occurrence of q and an occurrence of
            a in P, plus one.
            """
            def _get_indices_of_occurences(cand):
                def _flatten(l):
                    return [item for sublist in l for item in sublist]
                return np.array(_flatten(
                    [[i for i, x in enumerate(p) if x == num]
                        for index, num in enumerate(cand)])).reshape(-1, 1)
            question_indices = _get_indices_of_occurences(q)
            answer_indices = _get_indices_of_occurences(o)
            d = distance.cdist(question_indices, answer_indices, 'cityblock')
            return np.min(d) + 1

        o = [ele for ele in option if ele not in self.stopWords]
        p = [ele for ele in paragraph if ele not in self.stopWords]
        q = [ele for ele in question if ele not in self.stopWords]
        s_q = set(q) & set(p)
        s_a = (set(o) & set(p)) - set(q)
        if (len(s_q) == 0 or len(s_a) == 0):
            d = 1
        else:
            vals = []
            for q in s_q:
                for a in s_a:
                    d_p = distance_between(s_q, s_a, p)
                    vals.append(d_p)
            d = (1.0 / (len(p) - 1)) * min(vals)
        return d


    def PredictPerArticle(self, title, returnDict):
        '''
        @param returnDict: used to get return value from 
        different processes launched from multiprocessing
        '''
        print "Predicting for " + title
        # get unigram for the context
        article = self.data[title]
        contextUni = self.GetContextUnigram(article)
        # get all candidate span for the context
        contextSpan = self.GetContextConstituentSpan(article)
        # flatten version of contextUni
        context = []

        if self.articleLevel:
            for paraUni in contextUni:
                for sentenceUni in paraUni:
                    context += sentenceUni
        else:
            for paraUni in contextUni:
                contextPara = []
                for sentenceUni in paraUni:
                    contextPara += sentenceUni
                context.append(contextPara)

        # the questions are organized according to paragraph
        # but candidates are generated from the whole passage
        predictions = dict()
        for qaParaId, paragraph in enumerate(self.data[title].paragraphs):
            # predByPara = list()
            for qa in paragraph.qas:
                # pred = dict()
                preds = list()
                scores = list()
                # bestScore = sys.float_info.min
                qS = qa.question.sentence[0]
                qUnigram = [token.word.lower() for token in qS.token]
                if self.articleLevel:
                    for iPara, (para, unigrams, spans) \
                        in enumerate(zip(self.data[title].paragraphs, contextUni, contextSpan) ):
                        # traverse each sentence in the paragraph
                        for iSen, (s, uni, spanList) in enumerate(zip(para.context.sentence, unigrams, spans) ):
                            assert len(s.token) == len(uni)
                            for span in spanList:
                                beginId = span[0]
                                endId = span[1]
                                aUnigram = uni[beginId:endId]
                                
                                score = self.GetSlidingWindowScore(context, qUnigram, aUnigram) \
                                    - self.lambDist * self.GetMinDistance(context, qUnigram, aUnigram)
                                
                                scores.append(score)
                                ansStr = ReconstructStrFromSpan(s.token, span)
                                ansToken = s.token[span[0]:span[1] ]
                                preds.append(QaPrediction(title, qa.id, ansStr, iPara, iSen, ansToken=ansToken) )
                else:
                    iPara = qaParaId
                    para = self.data[title].paragraphs[iPara]
                    unigrams = contextUni[iPara]
                    spans = contextSpan[iPara]
                    for iSen, (s, uni, spanList) in enumerate(zip(para.context.sentence, unigrams, spans) ):
                        assert len(s.token) == len(uni)
                        for span in spanList:
                            beginId = span[0]
                            endId = span[1]
                            aUnigram = uni[beginId:endId]
                            
                            score = self.GetSlidingWindowScore(context, qUnigram, aUnigram) \
                                - self.lambDist * self.GetMinDistance(context, qUnigram, aUnigram)
                            
                            scores.append(score)
                            ansStr = ReconstructStrFromSpan(s.token, span)
                            ansToken = s.token[span[0]:span[1] ]
                            preds.append(QaPrediction(title, qaId, ansStr, iPara, iSen, ansToken=ansToken, score=score) )

                scores = np.array(scores)
                preds = np.array(preds)
                scoreOrder = np.argsort(-scores)
                predictions[qa.id] = preds[scoreOrder][0:min(self.topK, preds.size) ].tolist()
        returnDict[title] = predictions


    def Predict(self):
        print "Predicting!"
        returnDict = \
            MultipleProcess(agent=self, titleList=self.data.keys(),
            targetFunc=SlidingWindowAgent.PredictPerArticle)
        self.predictions = {}
        for title in returnDict.keys():
            self.predictions.update(returnDict[title], articleLevel=True, topK=1)


if __name__ == "__main__":
    dataFile = "../../archive/dev-anotated/dev-annotated.proto"
    articleLevel = True
    agent = SlidingWindowAgent(0, randSeed=0, articleLevel=articleLevel, topK=10)
    agent.LoadData(dataFile)
    agent.Predict()

    evalCandidateFile = "../../archive/dev-candidatesal/dev-candidatesal.proto"
    evalOrigFile = "../../archive/dev-anotated/dev-annotated.proto"
    vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/vocab_dict"
    sampleAgent = Agent(floatType=tf.float32, idType=tf.int32, lossType="max-margin", articleLevel=articleLevel)
    sampleAgent.LoadEvalData(evalCandidateFile, evalOrigFile)
    sampleAgent.LoadVocab(vocabPath)
    sampleAgent.PrepareData(doTrain=False)

    evaluator = QaEvaluator(wordToId=sampleAgent.wordToId, idToWord=sampleAgent.idToWord,
        metrics=("exact-match-top-1", "exact-match-top-3", "exact-match-top-5", 
        "in-sentence-rate-top-1", "in-sentence-rate-top-3", "in-sentence-rate-top-5",
        "<unk>-freq-in-pred") )
    evaluator.EvaluatePrediction(sampleAgent.evalSamples, agent.predictions)
    


# zip -r 1460521688980_new_dict.zip 1460521688980_new_dict
# cl upload 1460521688980_new_dict.zip

# rm src.zip
# zip -r src.zip src
# cl upload src.zip
# cl run qa-annotated-train-candidates-1460521688980_new.proto:0x400cc8/qa-annotated-train-candidates-1460521688980_new.proto qa-annotated-train-1460521688980_new.proto:0x757686/qa-annotated-train-1460521688980_new.proto _1460521688980_new_dict:_1460521688980_new_dict context_rnn.py:src/learning_baseline/context_rnn.py src:src 'python context_rnn.py' -n Bow-train --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3

# cl run qa-annotated-train-1460521688980_new.proto:0x757686/qa-annotated-train-1460521688980_new.proto data_processor.py:src/utils/data_processor.py src:src 'python data_processor.py' -n train_dict --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3

# rm src.zip vocab_dict.zip
# zip -r src.zip src
# zip -r vocab_dict.zip dataset/proto/vocab_dict
# cl upload src.zip
# cl upload vocab_dict.zip
# cl run train-annotated.proto:0x981923/train-annotated.proto train-candidatesal.proto:0x03fc46/train-candidatesal.proto dev-annotated.proto:0x753738/dev-annotated.proto dev-candidatesal.proto:0xdfa81b/dev-candidatesal.proto vocab_dict:vocab_dict src:src bow_context_rnn.py:src/learning_baseline/bow_context_rnn.py "python bow_context_rnn.py" -n bow_dim_100_xent_5e-2 --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john2
    


