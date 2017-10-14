import random
from spacy.en import English
import numpy as np
import string
from scipy.spatial import distance
from gensim.parsing.preprocessing import STOPWORDS


class Challenger(object):
    """Abstract class for a question answering agent."""

    def answer(self, paragraph, question, options):
        """Return the probability of the choices."""
        pass


class StaticGuesser(Challenger):
    """Guesses choice A."""

    def answer(self, paragraph, question, options):
        """Choose A."""
        a = np.zeros(len(options))
        a[0] = 1
        return a


class RandomGuesser(Challenger):
    """Guesses Randomly."""

    def answer(self, paragraph, question, options):
        """Choose a random number."""
        a = np.zeros(len(options))
        a[random.choice(range(len(options)))] = 1
        return a


class QuestionAnswerNeuralDistance(Challenger):
    """Looks at the distance between the q and choices."""

    def __init__(self):
        """Setup the spacy lang."""
        self.nlp = English()

    def answer(self, paragraph, question, options):
        """Choose the one with least distance between qa."""
        def extract_tokens(sent, ignore=False):
            if ignore:
                return [token for token in self.nlp(sent)]
            return [token for token in self.nlp(sent) if
                    not token.is_stop and not token.is_punct]
        q = extract_tokens(question)
        scores = []
        for option in options:
            opt = extract_tokens(option, ignore=True)
            dist = np.median([w1.similarity(w2)
                              for w1 in q for w2 in opt])
            scores.append(dist)
        return np.argmax(scores)


class SlidingWindow(Challenger):
    """Sliding window."""

    def transform_elem(self, s):
        """Get Tokens."""
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        return [elem for elem in s.lower().split() if elem not in STOPWORDS]

    def sw(self, paragraph, question, option):
        """Get the sliding window score."""
        def ic(w, p):
            return np.log(1 + (1.0 / p.count(w)))
        o = self.transform_elem(option)
        p = self.transform_elem(paragraph)
        q = self.transform_elem(question)
        s = set(o + q)
        scores = []
        for i in range(len(p)):
            sum = 0
            for j in range(len(s)):
                if i + j < len(p) and p[i + j] in s:
                    sum = sum + ic(p[i + j], p)
            scores.append(sum)
        return np.max(scores)

    def answer(self, paragraph, question, options):
        """Distance based answer."""
        scores = [self.sw(paragraph, question, option) for option in options]
        return np.exp(scores) / np.sum(np.exp(scores))


class SlidingWindowPlusDistanceBased(SlidingWindow):
    """Adds window."""

    def distance_between(self, q, o, p):
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

    def d(self, paragraph, question, option):
        """Distance based add to sliding."""
        o = self.transform_elem(option)
        p = self.transform_elem(paragraph)
        q = self.transform_elem(question)
        s_q = set(q) & set(p)
        s_a = (set(o) & set(p)) - set(q)
        if (len(s_q) == 0 or len(s_a) == 0):
            d = 1
        else:
            vals = []
            for q in s_q:
                for a in s_a:
                    d_p = self.distance_between(s_q, s_a, p)
                    vals.append(d_p)
            d = (1.0 / (len(p) - 1)) * min(vals)
        return d

    def answer(self, paragraph, question, options):
        """Distance based plus sliding answer."""
        scores = [self.sw(paragraph, question, option) -
                  self.d(paragraph, question, option) for option in options]
        return np.exp(scores) / np.sum(np.exp(scores))
Chat Conversation End
Type a message...



rm src.zip
zip -r src.zip src
cl upload src.zip
cl run data_processor.py:0x40c683/utils/data_processor.py qa-annotated-full-1460521688980_new.proto:0x4eb1eb/qa-annotated-full-1460521688980_new.proto src:0x40c683 'python data_processor.py' -n generate_dict --request-docker-image stanfordsquad/ubuntu:1.0 --request-queue john
