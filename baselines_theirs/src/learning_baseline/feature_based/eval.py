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
from learning_baseline.feature_based.compute_metrics import ComputeAndDisplayMetrics
from learning_baseline.feature_based.input import Dictionary, ReadExamples, GetInputPlaceholders, GetFeedDict, ReadQuestionAnnotations
from learning_baseline.feature_based.graph import GetLogits, GetVariables
from utils.squad_utils import ReconstructStrFromSpan

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'dataset/dev.json', '')
flags.DEFINE_string('input-articles', 'dataset/dev-annotatedpartial.proto', '')
flags.DEFINE_string('input-features', 'dataset/dev-featuresbucketized.proto', '')
flags.DEFINE_string('input-featuredict', 'dataset/featuredictbucketized.proto', '')
flags.DEFINE_string('input-model', 'dataset/model13-it3', '')
flags.DEFINE_integer('min-articles', None, '')
flags.DEFINE_boolean('print-errors', False, '')

if __name__ == '__main__':
    dictionary = Dictionary(FLAGS.input_featuredict, [])

    titles = set()
    examples = ReadExamples(FLAGS.input_features, dictionary, FLAGS.min_articles, titles)
    random.shuffle(examples)
    question_annotations = ReadQuestionAnnotations(FLAGS.input_articles)
    evaluator = Evaluator(FLAGS.input, titles)

    inputs = GetInputPlaceholders()
    variables = GetVariables(dictionary)
    logits = GetLogits(inputs, variables)    
    _, predict_op_top_1 = tf.nn.top_k(logits, 1)
    _, predict_op_top_3 = tf.nn.top_k(logits, 3)
                    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.input_model)
        
        ComputeAndDisplayMetrics(
            sess, inputs, None, predict_op_top_3, examples, question_annotations,
            evaluator, '')

        if FLAGS.print_errors:
            for example in examples:
                prediction, weights = sess.run([predict_op_top_1, variables.W], feed_dict=GetFeedDict(inputs, example))
                predictions = BuildPredictions(question_annotations, example, prediction)

                for question_index, question_id in enumerate(example.question_ids):
                    predicted_answer = predictions[question_id][0]
                    if not evaluator.ExactMatchSingle(question_id, predicted_answer):
                        annotations = question_annotations[question_id]
                        print 'Answers for:', annotations.article.title
                        print 'Question:', annotations.qa.question.text
                
                        if example.label[question_index] == prediction[question_index][0]:
                            print '  Correct Text (not a span):', evaluator._answers[question_id][0]
                            print '  Predicted Span:', predicted_answer
                        else:
                            correct_features = set()
                            predicted_features = set()
                            
                            for i in xrange(example.input_indices.shape[0]):
                                if example.input_indices[i][0] == question_index:
                                    if example.input_indices[i][1] == example.label[question_index]:
                                        correct_features.add(example.input_indices[i][2])
                                    elif example.input_indices[i][1] == prediction[question_index][0]:
                                        predicted_features.add(example.input_indices[i][2])
                            
                            same_features = correct_features & predicted_features
                            correct_features -= same_features
                            predicted_features -= same_features
                            
                            def PrintAnswer(candidate_index, features, prefix):
                                span = example.candidate_answers[candidate_index]
                                sentence_tokens = annotations.article.paragraphs[span.paragraphIndex].context.sentence[span.sentenceIndex].token
                                print '  ' + prefix + ' Sentence:',  ReconstructStrFromSpan(sentence_tokens)
                                print '  ' + prefix + ' Span:', BuildPrediction(annotations, span)
                
                                total_weight = 0
                                sorted_weights = []
                                for feature_index in features:
                                    total_weight += weights[feature_index]
                                    sorted_weights.append((weights[feature_index], dictionary.GetName(feature_index)))
                
                                print '  ' + prefix + ' Score:', total_weight
                                print '  ' + prefix + ' Features:'
                                sorted_weights.sort(reverse=True)
                                for weight, name in sorted_weights:
                                    print '    ' + str(weight), name
                
                            
                            PrintAnswer(example.label[question_index], correct_features, 'Correct')
                            print
                            PrintAnswer(prediction[question_index][0], predicted_features, 'Predicted')
                            print
                
                        correct_span = example.candidate_answers[example.label[question_index]]
                        predicted_span = example.candidate_answers[prediction[question_index][0]]
                        if correct_span.paragraphIndex == predicted_span.paragraphIndex and correct_span.sentenceIndex == predicted_span.sentenceIndex:
                            print 'Correct Sentence!'
                        else:
                            print 'Wrong Sentence!'
                
                        print
                        print
                        print
