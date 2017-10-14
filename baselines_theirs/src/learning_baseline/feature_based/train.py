import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import json
import logging
import random

import tensorflow as tf

from evaluation.evaluator import Evaluator
from learning_baseline.feature_based.compute_metrics import ComputeAndDisplayMetrics
from learning_baseline.feature_based.input import Dictionary, ReadExamples, ReadQuestionAnnotations, GetInputPlaceholders, GetFeedDict, FullDictionary
from learning_baseline.feature_based.graph import GetLogits, GetVariables


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input-train', '', '')
flags.DEFINE_string('input-train-articles', '', '')
flags.DEFINE_string('input-train-features', '', '')
flags.DEFINE_string('input-dev', '', '')
flags.DEFINE_string('input-dev-articles', '', '')
flags.DEFINE_string('input-dev-features', '', '')
flags.DEFINE_string('input-featuredict', '', '')
flags.DEFINE_boolean('use-full-dictionary', False, '')
flags.DEFINE_string('ablate-features', '', '')
flags.DEFINE_string('model-output', '', '')
flags.DEFINE_string('metrics-output', '', '')
flags.DEFINE_string('dev-predictions-output', '', '')
flags.DEFINE_integer('max-train-articles', None, '')
flags.DEFINE_integer('max-train-questions', 10000000, '')
flags.DEFINE_integer('max-dev-articles', None, '')
flags.DEFINE_integer('num-iterations', 1, '')
flags.DEFINE_float('learning-rate', 0.1, '')
flags.DEFINE_float('l2', 0.0, '')


logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(message)s',
                        level=logging.INFO)

    random.seed(123)

    dictionary = None
    if FLAGS.use_full_dictionary:
        dictionary = FullDictionary(FLAGS.input_featuredict, FLAGS.ablate_features.split(',') if FLAGS.ablate_features else [])
    else:
        dictionary = Dictionary(FLAGS.input_featuredict, FLAGS.ablate_features.split(',') if FLAGS.ablate_features else [])
        logger.info('Using %d features.', dictionary.NumFeatures())

    training_titles = set()
    training_examples = ReadExamples(FLAGS.input_train_features, dictionary, FLAGS.max_train_articles, training_titles)
    random.shuffle(training_examples) 

    dev_titles = set()
    dev_examples = ReadExamples(FLAGS.input_dev_features, dictionary, FLAGS.max_dev_articles, dev_titles)
    dev_question_annotations = ReadQuestionAnnotations(FLAGS.input_dev_articles)
    dev_evaluator = Evaluator(FLAGS.input_dev, dev_titles)

    # Use a small set of articles for computing the metrics on the training set.
    training_metric_titles = set(random.sample(training_titles, len(dev_titles))) if len(training_titles) > len(dev_titles) else training_titles
    training_metric_examples = [example for example in training_examples if example.article_title in training_metric_titles]
    training_question_annotations = ReadQuestionAnnotations(FLAGS.input_train_articles)
    training_evaluator = Evaluator(FLAGS.input_train, training_metric_titles)

    # Filter the training questions for the learning curve.
    num_training_questions = 0
    filtered_training_examples = []
    for example in training_examples:
        if example.num_questions + num_training_questions > FLAGS.max_train_questions:
            break
        num_training_questions += example.num_questions
        filtered_training_examples.append(example)
    training_examples = filtered_training_examples

    if FLAGS.use_full_dictionary:
        logger.info('Using %d features.', dictionary.NumFeatures())
    logger.info('Using %d training paragraphs and %s dev paragraphs', len(training_examples), len(dev_examples))

    inputs = GetInputPlaceholders()
    variables = GetVariables(dictionary)
    logits = GetLogits(inputs, variables)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputs.label))
    train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_op)
    _, predict_op = tf.nn.top_k(logits, k=3)
    
    scale_weights_op = variables.W.assign(variables.W * inputs.weight_scaling_constant)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        for it in xrange(1, FLAGS.num_iterations + 1):
            logger.info('Iteration %d', it)

            # Train.
            weight_scaling_constant = 1.0
            for i, example in enumerate(training_examples):
                if i != 0 and i % 1000 == 0:
                    logger.info('Training example %d', i)
                sess.run([train_op], feed_dict=GetFeedDict(inputs, example, weight_scaling_constant))
                weight_scaling_constant *= 1 - FLAGS.l2 / len(training_examples)
            sess.run([scale_weights_op], feed_dict={inputs.weight_scaling_constant: weight_scaling_constant})

            # Compute metrics.
            training_metrics, _ = ComputeAndDisplayMetrics(
                sess, inputs, loss_op, predict_op, training_metric_examples,
                training_question_annotations, training_evaluator, 'Training')
            dev_metrics, dev_predictions = ComputeAndDisplayMetrics(
                sess, inputs, loss_op, predict_op, dev_examples,
                dev_question_annotations, dev_evaluator, 'Dev')

            metrics = {}
            metrics['NumQuestions'] = num_training_questions
            metrics['NumFeatures'] = dictionary.NumFeatures()
            metrics['L2'] = FLAGS.l2
            metrics.update(training_metrics)
            metrics.update(dev_metrics)

            if FLAGS.ablate_features:
                metrics['AblateFeatures'] = FLAGS.ablate_features

            saver.save(sess, FLAGS.model_output + '-it' + str(it))
            with open(FLAGS.metrics_output + '-it' + str(it) + '.json', 'w') as f:
                f.write(json.dumps(metrics))
            with open(FLAGS.dev_predictions_output + '-it' + str(it) + '.json', 'w') as f:
                f.write(json.dumps(dev_predictions))
