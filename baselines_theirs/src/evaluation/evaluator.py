import json

class Evaluator(object):
    def __init__(self, path, restrict_to_titles=None):
        with open(path, 'r') as fileobj:
            articles = json.loads(fileobj.read())['data']
        self._answers = {}
        for article in articles:
            if restrict_to_titles is not None and article['title'] not in restrict_to_titles:
                continue
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    answers = self._answers[qa['id']] = []
                    for answer in qa['answers']:
                        answers.append(answer['text'])

    def ExactMatch(self, predictions, n=1):
        """
        Accepts a dict from question ID to a list of answers. The answers will
        be considered an exact match if any of the first n answers is an exact
        match.
        """
        total = 0
        for question_id in self._answers.iterkeys():
            predicted_answers = predictions.get(question_id)
            if predicted_answers is not None:
                for i in xrange(n):
                    if i >= len(predicted_answers):
                        break
                    if self.ExactMatchSingle(question_id, predicted_answers[i]):
                        total += 1
                        break

        return 100.0 * total / len(self._answers)

    def ExactMatchSingle(self, question_id, predicted_answer):
        predicted_answer = self.CleanAnswer(predicted_answer)
        for answer in self._answers[question_id]:
            if self.CleanAnswer(answer) == predicted_answer:
                return True
        return False

    IGNORED_WORDS = set(['the', 'a', 'an', 'in', 'to', 'over', 'by', 'between', 'at', 'after', 'from', 'as', 'for', 'around', 'about', 'on', 'since', 'through', 'with', 'within', 'if', 'of', 'before', 'during', 'near', 'under', 'although', 'because', 'out', 'above', 'into', 'towards', 'that', 'atop', 'besides', 'via', 'until', 'without'])

    def CleanAnswer(self, answer):
        answer = answer.lower()
        answer = answer.replace('\xc2\xa0', ' ')
        while len(answer) > 1 and answer[0] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
            answer = answer[1:]
        while len(answer) > 1 and answer[-1] in [' ', '.', ',', '!', ':', ';', '?', '`', '\'', '$']:
            answer = answer[:-1]

        answer_tokens = answer.split(' ')
        while len(answer_tokens) and answer_tokens[0] in Evaluator.IGNORED_WORDS:
            answer_tokens = answer_tokens[1:]
        
        return ' '.join(answer_tokens)
