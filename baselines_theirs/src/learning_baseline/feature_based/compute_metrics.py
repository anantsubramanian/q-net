import numpy as np

from learning_baseline.feature_based.build_prediction import BuildPredictions
from learning_baseline.feature_based.input import GetFeedDict

def ComputeAndDisplayMetrics(sess, inputs, loss_op, predict_top_3_op, examples, question_annotations, evaluator, metric_prefix):
    """
    Returns a tuple of metrics and predictions.
    """
    total_loss = 0
    total_in_sentence = 0
    total_in_sentence_top_3 = 0
    num_questions = 0
    output_predictions = {}
    for example in examples:
        ops = [predict_top_3_op]
        if loss_op is not None:
            ops.append(loss_op)
        res = sess.run(ops, feed_dict=GetFeedDict(inputs, example))
        predictions = res[0]

        if loss_op is not None:
            loss = res[1]
            if np.isnan(loss):
                print example.question_ids
                assert False
            total_loss += loss * example.num_questions

        num_questions += example.num_questions

        output_predictions.update(BuildPredictions(question_annotations, example, predictions))

        for question_index in xrange(example.num_questions):
            for i in xrange(3):
                prediced_span = example.candidate_answers[predictions[question_index][i]]
                correct_span = example.candidate_answers[example.label[question_index]]
                if (prediced_span.paragraphIndex == correct_span.paragraphIndex and
                    prediced_span.sentenceIndex == correct_span.sentenceIndex):
                    if i == 0:
                        total_in_sentence += 1
                    total_in_sentence_top_3 += 1
                    break

    metrics = {}
    if loss_op is not None:
        metrics[metric_prefix + 'Loss'] = total_loss / num_questions
        print 'Average', metric_prefix.lower(), 'loss', metrics[metric_prefix + 'Loss']
    metrics[metric_prefix + 'ExactMatch'] = evaluator.ExactMatch(output_predictions)
    print 'Average', metric_prefix.lower(), 'exact match:', metrics[metric_prefix + 'ExactMatch']
    metrics[metric_prefix + 'ExactMatchTop3'] = evaluator.ExactMatch(output_predictions, 3)
    print 'Average', metric_prefix.lower(), 'exact match top 3:', metrics[metric_prefix + 'ExactMatchTop3']
    metrics[metric_prefix + 'InSentence'] = 100.0 * total_in_sentence / num_questions
    print 'Average', metric_prefix.lower(), 'in sentence:', metrics[metric_prefix + 'InSentence']
    metrics[metric_prefix + 'InSentenceTop3'] = 100.0 * total_in_sentence_top_3 / num_questions
    print 'Average', metric_prefix.lower(), 'in sentence top 3:', metrics[metric_prefix + 'InSentenceTop3']
    
    return metrics, output_predictions
