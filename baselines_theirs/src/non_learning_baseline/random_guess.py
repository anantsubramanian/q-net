import numpy as np
import multiprocessing as mp
import random
import json
import sys
# import tensorflow as tf
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("./src/")
from proto import io
from utils.squad_utils import ReconstructStrFromSpan, ObjDict
from utils import squad_utils
from utils.squad_utils import LoadProtoData, DumpJsonPrediction
from utils.squad_utils import MultipleProcess
from utils.multiprocessor_cpu import MultiProcessorCPU
from evaluation_random.evaluator import Evaluator
from non_learning_baseline.non_learning_agent import NonLearningAgent
from learning_baseline.agent import Agent, QaPrediction


class RandGuessAgent(NonLearningAgent):
    '''
    Predict from stanford coreNLP constituent parsing results with
    the following rule:
    1. the constituent should not have overlapping words with
    the query excluding the stopwords.
    2. In each sentence, we check the complementary of the current
    constituent and score it with \lambda_uni * nContextOverlappingUnigram
    + \lambda_bi * nContextOverlappingBigram - \lambda_overlap * nAnswerOverlapUnigram.
    3. We pick the constituent with the highest score as the prediction.
    I.e. we encourage overlapping between query and answer context and 
    penalize overlapping between query and answer itself.
    '''
    def __init__(self, randSeed, articleLevel=True, topK=1):
        NonLearningAgent.__init__(self, randSeed)
        self.articleLevel = articleLevel
        self.topK = topK


    def PredictPerArticle(self, title, returnDict):
        '''
        @param returnDict: used to get return value from 
        different processes launched from multiprocessing
        '''
        # get unigram and bigram for the context
        article = self.data[title]
        contextUni = self.GetContextUnigram(article)
        contextBi = self.GetContextBigram(article)
        # get all candidate span for the context
        contextSpan = self.GetContextConstituentSpan(article)
        # the questions are organized according to paragraph
        # but candidates are generated from the whole passage
        predictions = dict()
        print "Predicting for " + title
        for qaParaId, paragraph in enumerate(self.data[title].paragraphs):
            # predByPara = list()
            for qa in paragraph.qas:
                pred = dict()
                bestScore = sys.float_info.min
                bestSentence = None
                qS = qa.question.sentence[0]
                qUnigram = [token.word.lower() for token in qS.token]
                qBigram = self.GetBigramBySentence(qS.token)
                # traverse over paragraphs
                pred = dict()
                pred["answer"] = []
                pred["token"] = []
                # if ansSentence not in pred["sentence"]:
                pred["sentence"] = []
                pred["sentenceToken"] = []
                pred["paraId"] = []
                pred["senId"] = []
                for iPara, (para, unigrams, bigrams, spans) \
                    in enumerate(zip(self.data[title].paragraphs, contextUni, contextBi, contextSpan) ):
                    # traverse each sentence in the paragraph
                    if self.articleLevel == False and iPara != qaParaId:
                        continue
                    for iSen, (s, uni, bi, spanList) in enumerate(zip(para.context.sentence, unigrams, bigrams, spans) ):
                        assert len(s.token) == len(uni)
                        assert len(s.token) == len(bi) + 1
                        for span in spanList:
                            beginId = span[0]
                            endId = span[1]
                            cbUnigram = uni[0:beginId]
                            caUnigram = uni[endId:]
                            if beginId == 0:
                                cbBigram = []
                            else:
                                cbBigram = bi[0:(beginId - 1) ]
                            caBigram = bi[endId:]
                            aUnigram = uni[beginId:endId]
                            aBigram = bi[beginId:(endId - 1) ]
                            # if len(aUnigram) == 1 and (aUnigram[0] == "." or aUnigram[0] == "?" or aUnigram[0] == "!" or aUnigram[0] == "the" or aUnigram[0] == "The"):
                            if len(aUnigram) == 1 and aUnigram[0] in self.stopWords:
                                continue
                            if len(aUnigram) > 3:
                                continue

                            ansStr = ReconstructStrFromSpan(s.token, span)
                            ansToken = s.token[span[0]:span[1] ]
                            ansSentence = ReconstructStrFromSpan(s.token, (0, len(s.token) ) )
                            ansSentenceToken = s.token[0:len(s.token) ]
                            assert len(s.token) != 0
                            # ansSentenceToken = [token.word for token in s.token]
                            pred["answer"].append(ansStr)
                            pred["token"].append(ansToken)
                            # if ansSentence not in pred["sentence"]:
                            pred["sentence"].append(ansSentence)
                            pred["sentenceToken"].append(ansSentenceToken)
                            pred["paraId"].append(iPara)
                            pred["senId"].append(iSen)
                # filter from the candidates for best choice
                preds = list()
                for ansToken, sentenceToken, ansStr, iPara, iSen \
                    in zip(pred["token"], pred["sentenceToken"], pred["answer"], pred["paraId"], pred["senId"] ):
                    ansToken = [token.word.lower() for token in ansToken]
                    if ansToken[-1] == ".":
                        ansToken = ansToken[:-1]
                    if ansToken[0] == "The" or ansToken[0] == "the":
                        ansToken = ansToken[1:]
                    if len(ansToken) == 0:
                        print " zero length ans token"
                    preds.append(QaPrediction(title, qa.id, ansStr, iPara, iSen, ansToken=ansToken) )
                random.shuffle(preds)

                predictions[qa.id] = preds[0:min(self.topK, len(preds) ) ]
                predictions[qa.id] = predictions[qa.id][0].ansStr
        returnDict[title] = predictions


    def Predict(self, debug=False):
        returnDict = \
            MultipleProcess(agent=self, titleList=self.data.keys(),
            targetFunc=RandGuessAgent.PredictPerArticle, debug=debug)
        self.predictions = {}
        for title in returnDict.keys():
            self.predictions.update(returnDict[title] )

                

if __name__ == "__main__":
    # dataFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    path = "../../archive/dev-anotated/"
    dataFile = path+"dev-annotated.proto"
    # predFile = "/Users/Jian/Data/research/squad/output/non-learning-baseline/uni-bi-1460521688980_new.predict"
    predFile = "./random-guess.pred"

    agent = RandGuessAgent(randSeed=0, articleLevel=False, topK=1)
    agent.LoadData(dataFile)
    agent.LoadStopWords()
    agent.Predict(debug=False)
    agent.DumpPrediction(predFile)

    # jsonDataFile = "/Users/Jian/Data/research/squad/dataset/json/dev.json"
    jsonDataFile = path + "../../data/dev-v1.0.json"
    evaluator = Evaluator(jsonDataFile)

    exactMatchRate = evaluator.ExactMatch(agent.predictions)
    F1 = evaluator.F1(agent.predictions)

    print "exact rate ", exactMatchRate
    print "F1 rate ", F1


    # evalCandidateFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-candidatesal.proto"
    # evalOrigFile = "/Users/Jian/Data/research/squad/dataset/proto/dev-annotated.proto"
    # vocabPath = "/Users/Jian/Data/research/squad/dataset/proto/vocab_dict"
    
    # evalCandidateFile = "./dev-candidatesal.proto"
    # evalOrigFile = "./dev-annotated.proto"
    # vocabPath = "./vocab_dict/proto/vocab_dict"

    # sampleAgent = Agent(floatType=tf.float32, idType=tf.int32, lossType="max-margin", articleLevel=agent.articleLevel)
    # sampleAgent.LoadEvalData(evalCandidateFile, evalOrigFile, doDebug=False)
    # sampleAgent.LoadVocab(vocabPath)
    # sampleAgent.PrepareData(doTrain=False)

    # evaluator = QaEvaluator(wordToId=sampleAgent.wordToId, idToWord=sampleAgent.idToWord,
    #     metrics=("exact-match-top-1", "exact-match-top-3", "exact-match-top-5", 
    #     "in-sentence-rate-top-1", "in-sentence-rate-top-3", "in-sentence-rate-top-5") )
    # evaluator.EvaluatePrediction(sampleAgent.evalSamples, agent.predictions)
 
# zip -r vocab_dict.zip dataset/proto/vocab_dict

# cl rm ^1-3
# rm src.zip vocab_dict
# zip -r src.zip src
# zip -r vocab_dict.zip dataset/proto/vocab_dict
# cl upload src.zip
# cl upload vocab_dict.zip

# cl run dev-annotated.proto:0x753738/dev-annotated.proto dev-candidatesal.proto:0xdfa81b/dev-candidatesal.proto vocab_dict:vocab_dict src:src random_guess.py:src/non_learning_baseline/random_guess.py "python random_guess.py" -n random_guess_latest --request-docker-image stanfordsquad/ubuntu:1.1 --request-queue host=john3
    


