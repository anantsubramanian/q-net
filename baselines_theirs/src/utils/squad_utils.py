import numpy as np
# import matplotlib.pyplot as plt
# import pylab
import re
import itertools
import json
import collections
import multiprocessing as mp
import random
import sys

sys.path.append("./src/")
from proto import io
from proto import CoreNLP_pb2
from proto import dataset_pb2
from proto import training_dataset_pb2
from utils.multiprocessor_cpu import MultiProcessorCPU

'''
some general pre/post processing tips:
1. should strip the space at the begining or end
2. consider the influence of punctuation at the end
3. be careful about empty string when using lib re functions
'''

def LoadJsonData(filePath):
    '''
    Load the file.
    @param filePath: filePath string
    '''
    with open(filePath) as dataFile:
        data = json.load(dataFile)
    return data


def LoadProtoData(filePath):
    data = io.ReadArticles(filePath)
    dataDict = dict()
    for article in data:
        title = article.title
        dataDict[title] = article
    return dataDict


def LoadCandidateData(dataFile):
    dataIn = io.ReadArticles(dataFile, cls=training_dataset_pb2.TrainingArticle)
    dataDict = dict()
    for data in dataIn:
        dataDict[data.title] = data
    return dataDict


def LoadOrigData(dataFile):
    dataIn = io.ReadArticles(dataFile, cls=dataset_pb2.Article)
    dataDict = dict()
    for data in dataIn:
        dataDict[data.title] = data
    return dataDict


def DumpJsonPrediction(filePath, predictions):
    '''
    currently only support top 1 prediction.
    the output put goes in the following format:
    {id : answer string}
    '''
    predDict = dict()
    for title in predictions.keys():
        for pred in predictions[title]:
            if len(pred["prediction"] ) == 0:
                continue
            predDict[pred["id"] ] = pred["prediction"][0]
    with open(filePath, "w") as outFile:
        json.dump(predDict, outFile)


def StripPunct(sentence):
    sentence = sentence.replace("...", "<elli>")
    if sentence[-1] == '.'\
        or sentence[-1] == '?' \
        or sentence[-1] == '!' \
        or sentence[-1] == ';' \
        or sentence[-1] == ",":
        sentence = sentence[:-1]
    sentence = sentence.replace("<elli>", "...")
    return sentence


def ParseJsonData(data):
    '''
    @param data is a json object. This is the version before 
    visualization functionality.
    ''' 
    dataPerArticle = dict()
    for article in data:
        text = ""
        # process articles to a list of sentences represented by list of words
        for paragraph in article["paragraphs"]:
            text += paragraph["context"].strip() + " "
        textInSentences = TextToSentence(text)
        queries = list()
        answers = list()
        qaIds = list()
        for paragraph in article["paragraphs"]:
            for qaPair in paragraph["qas"]:
                # turn everything into lower cases
                queries.append(StripPunct(qaPair["question"].lower().strip() ) )
                answers.append(StripPunct(qaPair["answer"].lower().strip() ) )
                qaIds.append(qaPair["id"] )
        dataPerArticle[article["title"] ] = { \
            "textInSentences": textInSentences,
            "queries": queries,
            "answers": answers,
            "qaIds": qaIds
        }
    return dataPerArticle


def TextToSentence(text):
    '''
    cut document into sentences with the given delimiters
    @param delimiters: delimiters to cut doc to sentences as a list of char
    @return sentences: list of full string of sentences
    '''
    caps = "([A-Z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co|Corp)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    numbers = "([-+]?)([0-9]+)(\.)([0-9]+)"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(caps + "[.] " + caps + "[.] " + caps + "[.] ","\\1<prd> \\2<prd> \\3<prd>",text)
    text = re.sub(caps + "[.] " + caps + "[.] ","\\1<prd> \\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    text = re.sub(numbers, "\\g<1>\\g<2><prd>\\g<4>", text)

    # # specific to current SQUAD dataset
    text = text.lower()
    suffixesSupp = "(\.)([a-z]+)"
    text = re.sub(suffixesSupp,"<prd>\\2",text)
    text = text.replace("...", "<elli>")
    text = text.replace("i.e.", "i<prd>e<prd>")
    text = text.replace("etc.", "etc<prd>")
    text = text.replace("u.s.", "u<prd>s<prd>")
    text = text.replace("v.s.", "v<prd>s<prd>")
    text = text.replace("vs.", "vs<prd>")
    text = text.replace(" v. ", " v<prd> ")
    text = text.replace("med.sc.d", "med<prd>sc<prd>d")
    text = text.replace("ecl.", "ecl<prd>")
    text = text.replace("hma.", "hma<prd>")
    text = text.replace("(r.", "(r<prd>")    # for some year related staff
    text = text.replace("(d.", "(d<prd>") 

    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    text = text.replace("<elli>", "...")
    sentences = text.split("<stop>")

    sentences = [s.strip() \
        for s in sentences if s.strip() != '']
    return sentences


def SentenceToWord(sentences):
    '''
    cut sentences to list of words
    @param sentences: a list of sentences
    @return sentencesInWords: a list containing list of words
    '''
    delimiters = "[ ,;\"\n\(\)]+"
    sentencesInWords = list()
    for sentence in sentences:
        sentence = StripPunct(sentence)
        sentence = sentence.replace("...", " ...")
        sentencesInWords.append(re.split(delimiters, sentence) )
        # omit the empty word produced by re.split
        sentencesInWords[-1] = [s.strip().lower() for s in sentencesInWords[-1] if s.strip() != '']
    return sentencesInWords


############### helper to multiprocess per article task with MultiprocessorCPU
def MultipleProcess(agent, titleList, targetFunc, conservative=True, debug=False):
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


################ helpers for protobuf based dataset#################
def ReconstructStrFromSpan(tokens, span=None):
    '''
    @param tokens: a protobuf object representing a list of tokens
    @param span: a pair (beginId, endId). Note endId is excluded. 
    '''
    if span is None:
        span = (0, len(tokens))
    string = ""
    beginId, endId = span
    for i in range(beginId, endId):
        string += tokens[i].word + tokens[i].after
    string = string.strip()
    return string


def GetContextBigram(article):
    '''
    article is an protobuf object for apecific article
    '''
    bigram = []
    for paragraph in article.paragraphs:
        bigramByPara = list()
        for s in paragraph.context.sentence:
            bigramByPara.append(GetBigramBySentence(s.token) )
        bigram.append(bigramByPara)
    return bigram


def GetContextUnigram(article):
    unigram = []
    for paragraph in article.paragraphs:
        unigramByPara = list()
        for s in paragraph.context.sentence:
            unigramBySentence = [token.word.lower() for token in s.token]
            unigramByPara.append(unigramBySentence)
        unigram.append(unigramByPara)
    return unigram


def GetBigramBySentence(tokens):
    '''
    tokens is a list of proto message object tokens
    '''
    bigram = []
    for i in range(len(tokens) - 1):
        bigram.append( (tokens[i].word.lower(), tokens[i + 1].word.lower() ) )
    return bigram


def GetContextConstituentSpan(article):
    '''
    @return span: the spans are organized by the following hierarchy
    span = [spanByPara1, spanByPara2, ...] Where
    spanByPara1 = [spanBySentence1, spanBySentence2, ...]
    spanBySentence1 is a list of spans extracted from the parsing tree
    '''
    span = []
    for paragraph in article.paragraphs:
        spanByPara = list()
        for s in paragraph.context.sentence:
            # tokens = [token.word for token in s.token]
            spanBySentence = GetConstituentSpanBySentence(s.parseTree)
            spanByPara.append(spanBySentence)
        span.append(spanByPara)
    return span


def GetConstituentSpanBySentence(parseTree):
    '''
    @param parseTree: a protobuf object
    extract span represented by nodes in the parsing trees
    '''
    def AddSpanToParseTree(parseTree, nextLeaf):
        '''
        @param parseTree: a protobuf object 
        fill in the yieldBeginIndex and yieldEndIndex fields for parsing trees
        '''
        if len(parseTree.child) == 0:
            parseTree.yieldBeginIndex = nextLeaf
            parseTree.yieldEndIndex = nextLeaf + 1
            return parseTree, nextLeaf + 1
        else:
            for i in range(len(parseTree.child) ):
                child, nextLeaf = \
                    AddSpanToParseTree(parseTree.child[i], nextLeaf)
                parseTree.child[i].CopyFrom(child)
            parseTree.yieldBeginIndex = parseTree.child[0].yieldBeginIndex
            parseTree.yieldEndIndex = parseTree.child[-1].yieldEndIndex
            return parseTree, nextLeaf

    parseTree, _ = AddSpanToParseTree(parseTree, nextLeaf=0)
    spans = list()
    visitList = list()
    visitList.append(parseTree)
    tokenList = list()
    while len(visitList) != 0:
        node = visitList.pop(0)
        spans.append( (node.yieldBeginIndex, node.yieldEndIndex) )
        for subTree in node.child:
            visitList.append(subTree)
    spansUniq = []
    [spansUniq.append(span) for span in spans if span not in spansUniq]
    return spansUniq


# some functions for debug
def GetCandidateAnsListInStr(candDataPerArticle, origDataPerArtice, ids, predId):
    '''
    for detailed use browse to prediction function of context rnn
    '''
    ansList = list()
    for idx in ids:
        predInfo = candDataPerArticle.candidateAnswers[idx]
        predParaId = predInfo.paragraphIndex
        predSenId = predInfo.sentenceIndex
        predSpanStart = predInfo.spanBeginIndex
        predSpanEnd = predInfo.spanBeginIndex + predInfo.spanLength
        tokens = origDataPerArticle.paragraphs[predParaId].context.sentence[predSenId].token[predSpanStart:predSpanEnd]
        predStr = ReconstructStrFromSpan(tokens, (0, len(tokens) ) )
        ansList.append(predStr)
    return ansList


# for serializing complex results 
def ObjDict(obj):
    return obj.__dict__


# display proto tokens
def PrintProtoToken(tokens):
    print [t.word for t in tokens]


# remove the and . from tokens
def StandarizeToken(tokens):
    if tokens[-1].word == ".":
        tokens = tokens[:-1]
    if len(tokens) > 0 and (tokens[0].word == "The" or tokens[0].word == "the"):
        tokens = tokens[1:]
    return tokens


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


def UnkrizeData(data, rate, padId, unkId):
    '''
    artificially set non-<pad> tokens to <unk>. The portion of 
    the artificial <unk> is indicated by rate. 
    '''
    mask = np.random.uniform(low=0.0, high=1.0, size=data.shape)
    mask = np.logical_and( (data != padId), (mask >= (1 - rate) ) )
    data[mask] = unkId
    return data







        






