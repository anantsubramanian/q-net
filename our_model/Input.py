import cPickle as pickle
import gzip
import json
import numpy
import string
import sys

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

class Dictionary:
  def __init__(self, lowercase=True, remove_punctuation=True,
               answer_start="ANSWERSTART", answer_end="ANSWEREND"):
    self.index_to_word = []
    self.word_to_index = {}
    self.mutable = True
    self.lowercase = lowercase
    self.remove_punctuation = remove_punctuation
    self.answer_start = answer_start
    self.answer_end = answer_end
    self.pad_index = self.add_or_get_index('<pad>')
    self.pos_tags = dict()

  def size(self):
    return len(self.index_to_word)

  def add_or_get_index(self, word):
    if word == self.answer_start or word == self.answer_end:
      return -1
    # We ignore punctuation symbols
    if self.remove_punctuation:
      if word in string.punctuation:
        return None
    if self.lowercase:
      word = word.strip().lower()
    if word in self.word_to_index:
      return self.word_to_index[word]
    if not self.mutable:
      return self.pad_index
    new_index = len(self.index_to_word)
    self.word_to_index[word] = new_index
    self.index_to_word.append(word)
    return new_index

  def add_or_get_tag(self, tag):
    if tag in self.pos_tags:
      return self.pos_tags[tag]
    self.pos_tags[tag] = len(self.pos_tags)
    return self.pos_tags[tag]

  def get_index(self, word):
    if word == self.answer_start or word == self.answer_end:
      return -1
    if self.remove_punctuation:
      if word in string.punctuation:
        return None
    if word in self.word_to_index:
      return self.word_to_index[word]
    return self.pad_index

  def get_word(self, index):
    return self.index_to_word[index]

  def set_immutable(self):
    self.mutable = False


class Data:
  def __init__(self, dictionary=None, immutable=False):
    self.dictionary = Dictionary(lowercase=False,
                                 remove_punctuation=False)
    if dictionary:
      self.dictionary = dictionary
    if immutable:
      self.dictionary.set_immutable()
    self.sentence_end_markers = [ ".", "...", "!", "?", ";" ]
    self.questions = {}
    self.paragraphs = []
    self.tokenized_paras = []
    self.paras_tags = []
    self.question_to_paragraph = {}
    self.data = []
    self.missed = 0

  def clear_aux_data(self):
    del self.questions
    del self.paragraphs
    del self.question_to_paragraph
    del self.tokenized_paras
    del self.data

  def dump_pickle(self, filename):
    with gzip.open(filename + ".gz", 'wb') as fout:
      pickle.dump(self, fout)
      fout.close()

  def read_from_pickle(self, filename):
    with gzip.open(filename + ".gz", 'rb') as fin:
      self = pickle.load(fin)
      fin.close()
      return self

  def tokenize_para(self, para_text):
    # Create tokenized paragraph representation.
    tokenized_para = [ [ self.dictionary.add_or_get_index(word) \
                         for word in word_tokenize(sent) ] \
                           for sent in sent_tokenize(para_text) ]
    tokenized_para = [ filter(None, x) for x in tokenized_para ]
    tokenized_para = [ x for x in tokenized_para if len(x) > 0 ]
    joined_para = []
    map(joined_para.extend, tokenized_para)

    sentences = len(tokenized_para)
    if sentences == 0:
      print "Found no sentences for para:", para_text
      return None

    return joined_para

  def word_tokenize_para(self, para_text):
    # Create tokenized paragraph representation.
    tokenized_para_words = word_tokenize(para_text)
    tokenized_para = [ self.dictionary.add_or_get_index(word) \
                         for word in tokenized_para_words ]
    return tokenized_para, tokenized_para_words

  def f1_score(self, start, end, ans_start_idx, ans_end_idx):
    # Get the F1 score for two given ranges: candidate range and true range.
    if end < start:
      return 0.0
    intersection = min(end, ans_end_idx) - max(start, ans_start_idx)
    if intersection < 0:
      return 0.0
    intersection += 1
    true = ans_end_idx - ans_start_idx + 1
    ours = end - start + 1
    precision = intersection/float(ours)
    recall = intersection/float(true)
    return 2 * precision * recall / (precision + recall)

  def get_sent_start_end(self, tokenized_para, ans_start_idx, ans_end_idx):
    start_idx = ans_start_idx
    end_idx = ans_end_idx
    while start_idx > 0 and \
          tokenized_para[start_idx] not in self.sentence_end_markers:
      start_idx -= 1
    while end_idx < len(tokenized_para) and \
          tokenized_para[end_idx] not in self.sentence_end_markers:
      end_idx += 1

    # Ignore the punctuation if necessary.
    if not start_idx == ans_start_idx:
      start_idx += 1
    if not end_idx == ans_end_idx:
      end_idx -= 1

    return start_idx, end_idx

  def add_paragraph(self, paragraph):
    para_text = paragraph['context']
    para_qas = paragraph['qas']

    # Store tokenized paragraph.
    tokenized_para, tokenized_para_words = self.word_tokenize_para(para_text)
    self.tokenized_paras.append(tokenized_para)
    _, para_tags = zip(*pos_tag(tokenized_para_words))
    para_tags = [ self.dictionary.add_or_get_tag(tag) for tag in para_tags ]
    self.paras_tags.append(para_tags)

    for qa in para_qas:
      # Questions of length <= 2 words are ignored
      if len(word_tokenize(qa['question'])) <= 2:
        continue
      self.question_to_paragraph[qa['id']] = len(self.paragraphs)
      self.questions[qa['id']] = qa['question']

      # Tokenize question
      tokenized_question = word_tokenize(qa['question'])
      processed_question = [ self.dictionary.add_or_get_index(word) \
                               for word in tokenized_question ]
      processed_question = filter(None, processed_question)
      _, question_tags = zip(*pos_tag(tokenized_question))
      question_tags = [ self.dictionary.add_or_get_tag(tag) for tag in question_tags ]

      # Tokenize answer phrases
      processed_answers = []
      sentence_idxs = []
      for answer in qa['answers']:
        start_idx = answer['answer_start']
        end_idx = start_idx + len(answer['text'])
        para_text_modified = para_text[:start_idx] + " " + \
                             self.dictionary.answer_start + " " + \
                             para_text[start_idx:end_idx] + " " + \
                             self.dictionary.answer_end + " " + \
                             para_text[end_idx:]
        answer_idxs = [ i for i,idx in \
                          enumerate(self.word_tokenize_para(para_text_modified)[0]) \
                          if idx == -1 ]
        answer_idxs[1] -= 2

        # Valid answers should lie within bounds.
        if answer_idxs[0] < 0 or answer_idxs[0] >= len(tokenized_para) \
           or answer_idxs[1] < 0 or answer_idxs[1] >= len(tokenized_para):
          print "\n" * 3
          print "Invalid answer \"%s\" ignored. (%d,%d)\n" % \
                (answer['text'], answer_idxs[0], answer_idxs[1])
          print "Question: %s\n" % qa['question']
          print "Paragraph: %s" % para_text
          print "\n" * 3
          self.missed += 1
          continue

        processed_answers.append(answer_idxs)

        # Store paragraph sentence start and end indexes for input.
        sentence_idxs.append(self.get_sent_start_end(tokenized_para_words,
                                                     answer_idxs[0],
                                                     answer_idxs[1]))
        assert sentence_idxs[-1][0] >= 0 and \
               sentence_idxs[-1][0] < len(tokenized_para_words) and \
               sentence_idxs[-1][0] <= answer_idxs[0]
        assert sentence_idxs[-1][1] >= sentence_idxs[-1][0] and \
               sentence_idxs[-1][1] < len(tokenized_para_words) and \
               sentence_idxs[-1][1] >= answer_idxs[1]

      # Create question-answer pairs
      for processed_answer, sentence_idx in zip(processed_answers, sentence_idxs):
        f1_partial_matrix = numpy.zeros((processed_answer[1]+1, len(tokenized_para)-processed_answer[0]))
        for start in range(0,processed_answer[1]+1):
          for end in range(max(start,processed_answer[0]),len(tokenized_para)):
            f1_partial_matrix[start, end-processed_answer[0]] = \
              self.f1_score(start, end, processed_answer[0], processed_answer[1])
        self.data.append([processed_question, processed_answer, qa['id'], f1_partial_matrix,
                          question_tags, sentence_idx])

    # Store paragraph text
    self.paragraphs.append(para_text)


  def read_from_file(self, filename, max_articles):
    dev_data = {}
    with open(filename, 'r') as input_file:
      data = json.load(input_file)
      dev_data['version'] = data['version']
      dev_data['data'] = []

      data = data['data']
      # Read each input article
      for article_index, article in enumerate(data):
        if article_index == max_articles:
          break
        dev_data['data'].append(article)
        # Read each para for each article
        for para_index, paragraph in enumerate(article['paragraphs']):
          self.add_paragraph(paragraph)
          print "\r%d Articles, %d Paragraphs processed." \
                  % (article_index+1, para_index+1),
          sys.stdout.flush()
      print ""

    return dev_data

# Pad a given sequence upto length "length" with the given "element".
def pad(seq, element, length):
  assert len(seq) <= length
  padded_seq = seq + [element] * (length - len(seq))
  assert len(padded_seq) == length
  return padded_seq

# Create a 2D numpy array of shape "length"x"length" with the given values set,
# and rest set to element.
def create2d(partial_matrix, element, length, ans_start):
  padded_mat = numpy.zeros((length, length))
  padded_mat[:partial_matrix.shape[0], ans_start:ans_start+partial_matrix.shape[1]] = partial_matrix
  return padded_mat

# Create a one-hot vector of the given size, with position 'pos' set to 1.
def one_hot(pos, size):
  return [ 1 if i == pos else 0 for i in range(size) ]

# Read train and dev data, either from json files or from pickles, and dump them in
# pickles if necessary.
def read_data(train_json, train_pickle, dev_json, dev_pickle, max_train_articles,
              max_dev_articles, dump_pickles):
  reload(sys)
  sys.setdefaultencoding('utf-8')
  train_data = Data()
  print "Reading training data."
  if train_json:
    train_data.read_from_file(train_json, max_train_articles)
  else:
    train_data = train_data.read_from_pickle(train_pickle)

  dev_data = Data(train_data.dictionary)
  if dev_json:
    print "Reading dev data."
    dev_json_data = dev_data.read_from_file(dev_json, max_dev_articles)
  else:
    print "Reading dev data."
    dev_data = dev_data.read_from_pickle(dev_pickle)
    print "Done."

  if dump_pickles:
    assert not train_pickle == None
    assert not dev_pickle == None
    print "Dumping pickles."
    train_data.dump_pickle(train_pickle)
    dev_data.dump_pickle(dev_pickle)
    print "Done."

  print "Finished reading all required data."
  print "Train missed %d questions, Dev missed %d." % (train_data.missed, dev_data.missed)

  print "Done."
  print "Train vocab size is:", train_data.dictionary.size()
  print "Dev vocab size is:", dev_data.dictionary.size()
  print "Sample words:", train_data.dictionary.index_to_word[:10]

  return train_data, dev_data

