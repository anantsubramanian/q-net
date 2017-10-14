def BuildPredictions(question_annotations, example, top_predicted_candidate_indices):
    predictions = {}
    for question_index, question_id in enumerate(example.question_ids):
        question_predictions = predictions[question_id] = []
        for i in xrange(top_predicted_candidate_indices.shape[1]):
            predicted_span = example.candidate_answers[top_predicted_candidate_indices[question_index][i]]
            question_predictions.append(BuildPrediction(question_annotations[question_id], predicted_span))
    return predictions

def BuildPrediction(annotations, predicted_span):
    sentence_tokens = annotations.article.paragraphs[predicted_span.paragraphIndex].context.sentence[predicted_span.sentenceIndex].token
    predicted_answer = ''
    for i in xrange(predicted_span.spanBeginIndex, predicted_span.spanBeginIndex + predicted_span.spanLength):
        predicted_answer += sentence_tokens[i].originalText
        if i != predicted_span.spanBeginIndex + predicted_span.spanLength - 1:
            predicted_answer += sentence_tokens[i].after
    return predicted_answer
