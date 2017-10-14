import cPickle as pickle
import json
import numpy
import sys

from nltk.tokenize import sent_tokenize, word_tokenize

class Dictionary:
  def __init__(self, lowercase=True):
    self.index_to_word = []
    self.word_to_index = {}
    self.mutable = True
    self.lowercase = lowercase
    self.unk_index = self.add_or_get_index('<unk>')

  def size(self):
    return len(self.index_to_word)

  def add_or_get_index(self, word):
    if self.lowercase:
      word = word.strip().lower()
    if word in self.word_to_index:
      return self.word_to_index[word]
    if not self.mutable:
      return self.unk_index
    new_index = len(self.index_to_word)
    self.word_to_index[word] = new_index
    self.index_to_word.append(word)
    return new_index

  def get_index(self, word):
    if word in self.word_to_index:
      return self.word_to_index[word]
    return self.unk_index

  def get_word(self, index):
    return self.index_to_word[index]

  def set_immutable(self):
    self.mutable = False


class Data:
  def __init__(self, dictionary=None, num_incorrect_candidates=None):
    self.dictionary = Dictionary()
    if dictionary:
      self.dictionary = dictionary
      self.dictionary.set_immutable()
    self.questions = {}
    self.paragraphs = []
    self.question_to_paragraph = {}
    self.data = []
    self.num_incorrect_candidates = num_incorrect_candidates

  def dump_pickle(self, filename):
    with open(filename, 'wb') as fout:
      pickle.dump(self, fout)
      fout.close()

  def read_from_pickle(self, filename):
    with open(filename, 'rb') as fin:
      self = pickle.load(fin)
      fin.close()
      return self

  def add_paragraph(self, paragraph):
    para_text = paragraph['context']
    para_qas = paragraph['qas']

    for word in word_tokenize(para_text):
      self.dictionary.add_or_get_index(word)

    seen_answers = set()
    for qa in para_qas:
      self.question_to_paragraph[qa['id']] = len(self.paragraphs)
      self.questions[qa['id']] = qa['question']
      # Tokenize question
      processed_question = [ self.dictionary.add_or_get_index(word) \
                               for word in word_tokenize(qa['question']) ]
      # Tokenize answer phrases
      processed_answers = []
      correct_answers = set()
      for answer in qa['answers']:
        answer = [ self.dictionary.add_or_get_index(word) \
                     for word in word_tokenize(answer['text']) ]
        correct_answers.add(",".join(map(str,answer)))
        processed_answers.append(answer)

      # Create question-answer pairs
      for processed_answer in processed_answers:
        self.data.append([processed_question, processed_answer, 1.0])
      
      # If num_candidate is provided, sample those many incorrect candidate answers
      # from a paragraph, for each question.
      # Otherwise, generate all ~L(L-1)/2 incorrect candidates.
      tokenized_para = [ self.dictionary.get_index(word) \
                           for word in word_tokenize(para_text) ]
      para_len = len(tokenized_para)
      candidate_answers = []
      if self.num_incorrect_candidates:
        while len(candidate_answers) < self.num_incorrect_candidates:
          start_index = numpy.random.randint(0, para_len)
          end_index = numpy.random.randint(start_index+1, para_len+1)
          candidate = tokenized_para[start_index:end_index]
          candidate_str = ",".join(map(str, candidate))
          if not candidate_str in seen_answers and not candidate_str in correct_answers:
              candidate_answers.append(candidate)
              seen_answers.add(candidate_str)
      else:
        for start_index in range(len(tokenized_para)):
          for end_index in range(start_index+1, len(tokenized_para)):
            candidate_answer = tokenized_para[start_index:end_index]
            if ",".join(map(str,candidate_answer)) in correct_answers:
              continue
            candidate_answers.append(candidate_answer)

      for candidate_answer in candidate_answers:
        self.data.append([processed_question, candidate_answer, 0.0])

  def read_from_file(self, filename, max_articles):
    with open(filename, 'r') as input_file:
      data = json.load(input_file)
      data = data['data']

      # Read each input article paragraph
      for article_index, article in enumerate(data):
        if article_index == max_articles:
          break
        # Read each para for each article
        for para_index, paragraph in enumerate(article['paragraphs']):
          self.add_paragraph(paragraph)
          print "\r%d Articles, %d Paragraphs processed." \
                  % (article_index+1, para_index+1),
          sys.stdout.flush()
      print ""

