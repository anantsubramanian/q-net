import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import random
import tensorflow as tf
from tensorflow import flags

from evaluation.evaluator import Evaluator
from learning_baseline.feature_based.build_prediction import BuildPrediction, BuildPredictions
from learning_baseline.feature_based.input import Dictionary, ReadExamples, GetInputPlaceholders, GetFeedDict, ReadQuestionAnnotations
from learning_baseline.feature_based.graph import GetLogits, GetVariables
from utils.squad_utils import ReconstructStrFromSpan

FLAGS = flags.FLAGS
flags.DEFINE_string('input-articles', 'dataset/dev-annotatedpartial.proto', '')
flags.DEFINE_string('input-features', 'dataset/dev-featuresbucketized.proto', '')
flags.DEFINE_string('input-featuredict', 'dataset/featuredictbucketized-25000.proto', '')
flags.DEFINE_integer('min-articles', None, '')

if __name__ == '__main__':
    dictionary = Dictionary(FLAGS.input_featuredict)
    feature_index = dictionary.GetIndex('Dep Path NN - conj -> NN')

    examples = ReadExamples(FLAGS.input_features, dictionary, FLAGS.min_articles)
    question_annotations = ReadQuestionAnnotations(FLAGS.input_articles)

    for example in examples:
        for i in xrange(example.input_indices.shape[0]):
            question_index = example.input_indices[i][0]
            if example.input_indices[i][2] == feature_index:
                correct = example.input_indices[i][1] == example.label[question_index]

                annotations = question_annotations[example.question_ids[question_index]]
                span = example.candidate_answers[example.input_indices[i][1]]
                sentence_tokens = annotations.article.paragraphs[span.paragraphIndex].context.sentence[span.sentenceIndex].token
                print 'Sentence:',  ReconstructStrFromSpan(sentence_tokens)
                print 'Question:', annotations.qa.question.text
                print 'Span:', BuildPrediction(annotations, span)
                print 'Correct!' if correct else 'Wrong!'
                print
                
