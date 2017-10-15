import cPickle as pickle
import json
import numpy
import string
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
    # We ignore punctuation symbols
    if word in string.punctuation:
      return None
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
    if word in string.punctuation:
      return None
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

  def clear_aux_data(self):
    del self.questions
    del self.paragraphs
    del self.question_to_paragraph
    del self.data

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

    seen_answers = set()
    for qa in para_qas:
      # Questions of length <= 2 words are ignored
      if len(word_tokenize(qa['question'])) <= 2:
        continue
      self.question_to_paragraph[qa['id']] = len(self.paragraphs)
      self.questions[qa['id']] = qa['question']

      # Tokenize question
      processed_question = [ self.dictionary.add_or_get_index(word) \
                               for word in word_tokenize(qa['question']) ]
      processed_question = filter(None, processed_question)

      # Tokenize answer phrases
      processed_answers = []
      correct_answers = set()
      for answer in qa['answers']:
        answer = [ self.dictionary.add_or_get_index(word) \
                     for word in word_tokenize(answer['text']) ]
        answer = filter(None, answer)
        correct_answers.add(",".join(map(str,answer)))
        processed_answers.append(answer)

      # Create question-answer pairs
      for processed_answer in processed_answers:
        self.data.append([processed_question, processed_answer, 1.0, qa['id']])

      # If num_candidate is provided, sample those many incorrect candidate answers
      # from a paragraph, for each question.
      # Otherwise, generate all ~L(L-1)/2 incorrect candidates for each sentence.
      tokenized_para = [ [ self.dictionary.add_or_get_index(word) \
                           for word in word_tokenize(sent) ] \
                             for sent in sent_tokenize(para_text) ]
      tokenized_para = [ filter(None, x) for x in tokenized_para ]
      tokenized_para = [ x for x in tokenized_para if len(x) > 0 ]
      sentences = len(tokenized_para)
      if sentences == 0:
        print "Found no sentences for para:", para_text
        continue
      sent_lens = [ len(x) for x in tokenized_para ]

      # If the number of incorrect candidates to sample is set, use sampling.
      # Otherwise, generate ~L(L-1)/2 candidate answers.
      if self.num_incorrect_candidates:
        candidate_answers = []
        while len(candidate_answers) < self.num_incorrect_candidates:
          sent_index = numpy.random.randint(0, sentences)
          start_index = numpy.random.randint(0, sent_lens[sent_index])
          end_index = numpy.random.randint(start_index+1, sent_lens[sent_index] + 1)
          candidate = tokenized_para[sent_index][start_index:end_index]
          candidate_str = ",".join(map(str, candidate))
          if not candidate_str in seen_answers and not candidate_str in correct_answers:
              candidate_answers.append(candidate)
              seen_answers.add(candidate_str)
        for candidate_answer in candidate_answers:
          self.data.append([processed_question, candidate_answer, 0.0, qa['id']])
      else:
        seen = set()
        for sent in tokenized_para:
          for start_index in range(len(sent)):
            for end_index in range(start_index+1, len(sent)):
              candidate_answer = sent[start_index:end_index]
              candidate_str = ",".join(map(str,candidate_answer))
              if candidate_str in correct_answers or candidate_str in seen:
                continue
              self.data.append([processed_question, candidate_answer, 0.0, qa['id']])
              seen.add(candidate_str)

    # Store paragraph text
    self.paragraphs.append(para_text)


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

def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r

def read_data(train_json, train_pickle, dev_json, dev_pickle, test_pickle,
              num_incorrect_samples, max_train_articles, max_dev_articles,
              dump_pickles, load_test=False):
  train_data = Data(num_incorrect_candidates=num_incorrect_samples)
  print "Reading training data."
  if train_json:
    train_data.read_from_file(train_json, max_train_articles)
    if dump_pickles:
      assert not train_pickle == None
      dump_pickle(train_pickle)
  else:
    train_data = train_data.read_from_pickle(train_pickle)

  print "Done."
  print "Vocab size is:", train_data.dictionary.size()
  print "Sample words:", train_data.dictionary.index_to_word[:10]

  dev_data = Data(train_data.dictionary)
  test_data = Data(train_data.dictionary)
  print "Reading dev/test data."
  if dev_json:
    dev_data.read_from_file(dev_json, max_dev_articles)

    # Batch together data points by paragraph id
    para_to_examples = []
    for example in dev_data.data:
      if dev_data.question_to_paragraph[example[3]] >= len(para_to_examples):
        para_to_examples.append([])
      para_to_examples[dev_data.question_to_paragraph[example[3]]].append(example)
    new_dev_data = []

    # First 200 paragraphs are dev data, remaining are test data
    for para in para_to_examples[:200]:
      for example in para:
        new_dev_data.append(example)
    new_test_data = []
    for para in para_to_examples[200:]:
      for example in para:
        new_test_data.append(example)

    # Create test data
    test_data = Data(dev_data.dictionary)
    test_data.questions = dev_data.questions
    test_data.question_to_paragraph = dev_data.question_to_paragraph
    test_data.num_incorrect_candidates = dev_data.num_incorrect_candidates
    test_data.paragraphs = dev_data.paragraphs[200:]
    dev_data.paragraphs = dev_data.paragraphs[:200]
    dev_data.data = new_dev_data
    test_data.data = new_test_data

    if dump_pickles:
      assert not dev_pickle == None and not test_pickle == None
      dev_data.dump_pickle(dev_pickle)
      test_data.dump_pickle(test_pickle)
  else:
    dev_data = dev_data.read_from_pickle(dev_pickle)
    if load_test:
      test_data = test_data.read_from_pickle(test_pickle)
  print "Done."

  return train_data, dev_data, test_data

