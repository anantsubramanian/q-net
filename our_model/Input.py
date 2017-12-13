import corenlp
import cPickle as pickle
import gzip
import json
import numpy
import string
import sys
import time

from joblib import Parallel, delayed
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

# Define URL of running StanfordCoreNLPServer.
corenlp_url = 'http://localhost:9001'
max_tries = 10

def word_tokenize(idx, sentence):
  stanford_corenlp = StanfordCoreNLP(corenlp_url)
  tries = 0
  while True:
    try:
      annotation = stanford_corenlp.annotate(
        sentence.encode('utf8'),
        properties = { 'annotators': 'tokenize',
                       'outputFormat': 'json' })
      assert type(annotation) == dict
      break
    except Exception:
      time.sleep(1)
      tries += 1
      if tries == 10:
        return (idx, None)
      pass
  tokens = []
  tokens.extend([ token['word'] for token in annotation['tokens'] ])
  return (idx, tokens)

def tokenize_and_tag(idx, sentence):
  stanford_corenlp = StanfordCoreNLP(corenlp_url)
  tries = 0
  while True:
    try:
      annotation = stanford_corenlp.annotate(
        sentence.encode('utf8'),
        properties = { 'annotators': 'tokenize,pos,ner',
                       'outputFormat': 'json' })
      assert type(annotation) == dict
      break
    except Exception:
      time.sleep(1)
      tries += 1
      if tries == 10:
        print "Failed for %s" % sentence
        return (idx, None, None, None)
      pass
  tokens, pos_tags, ner_tags = [], [], []
  for sentence in annotation['sentences']:
    tokens.extend([ token['word'] for token in sentence['tokens'] ])
    pos_tags.extend([ token['pos'] for token in sentence['tokens'] ])
    ner_tags.extend([ token['ner'] for token in sentence['tokens'] ])
  return (idx, tokens, pos_tags, ner_tags)

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
    self.ner_tags = dict()

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

  def add_or_get_postag(self, pos_tag):
    if pos_tag in self.pos_tags:
      return self.pos_tags[pos_tag]
    self.pos_tags[pos_tag] = len(self.pos_tags)
    return self.pos_tags[pos_tag]

  def add_or_get_nertag(self, ner_tag):
    if ner_tag in self.ner_tags:
      return self.ner_tags[ner_tag]
    self.ner_tags[ner_tag] = len(self.ner_tags)
    return self.ner_tags[ner_tag]

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


def get_sent_start_end(tokenized_para, ans_start_idx, ans_end_idx):
  sentence_end_markers = [ ".", "...", "!", "?", ";" ]
  start_idx = ans_start_idx
  end_idx = ans_end_idx
  while start_idx > 0 and \
        tokenized_para[start_idx] not in sentence_end_markers:
    start_idx -= 1
  while end_idx < len(tokenized_para) and \
        tokenized_para[end_idx] not in sentence_end_markers:
    end_idx += 1

  # Ignore the punctuation if necessary.
  if not start_idx == ans_start_idx:
    start_idx += 1
  if not end_idx == ans_end_idx:
    end_idx -= 1

  return start_idx, end_idx

def f1_score(start, end, ans_start_idx, ans_end_idx):
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

# Create question-answer tuple with required information.
def create_data(qid, para_text, tokenized_para, tokenized_para_words,
                processed_question, dictionary, question, answers):
  missed = 0
  processed_answers = []
  sentence_idxs = []
  for answer in answers:
    start_idx = answer['answer_start']
    end_idx = start_idx + len(answer['text'])
    para_text_modified = para_text[:start_idx] + " " + \
                         dictionary.answer_start + " " + \
                         para_text[start_idx:end_idx] + " " + \
                         dictionary.answer_end + " " + \
                         para_text[end_idx:]
    answer_idxs = \
      [ i for i,idx in \
          enumerate([ dictionary.get_index(w) for w in \
                        word_tokenize(None, para_text_modified)[1] ]) \
          if idx == -1 ]
    answer_idxs[1] -= 2

    # Valid answers should lie within bounds.
    if answer_idxs[0] < 0 or answer_idxs[0] >= len(tokenized_para) \
       or answer_idxs[1] < 0 or answer_idxs[1] >= len(tokenized_para):
      print "\n" * 3
      print "Invalid answer \"%s\" ignored. (%d,%d)\n" % \
            (answer['text'], answer_idxs[0], answer_idxs[1])
      print "Question: %s\n" % question
      print "Paragraph: %s" % para_text
      print "\n" * 3
      missed += 1
      continue

    processed_answers.append(answer_idxs)

    # Store paragraph sentence start and end indexes for input.
    sentence_idxs.append(get_sent_start_end(tokenized_para_words,
                                            answer_idxs[0],
                                            answer_idxs[1]))
    assert sentence_idxs[-1][0] >= 0 and \
           sentence_idxs[-1][0] < len(tokenized_para_words) and \
           sentence_idxs[-1][0] <= answer_idxs[0]
    assert sentence_idxs[-1][1] >= sentence_idxs[-1][0] and \
           sentence_idxs[-1][1] < len(tokenized_para_words) and \
           sentence_idxs[-1][1] >= answer_idxs[1]

  # Create question-answer tuples.
  data = []
  for processed_answer, sentence_idx in zip(processed_answers, sentence_idxs):
    f1_partial_matrix = numpy.zeros((processed_answer[1]+1,
                                     len(tokenized_para)-processed_answer[0]))
    for start in range(0,processed_answer[1]+1):
      for end in range(max(start,processed_answer[0]),len(tokenized_para)):
        f1_partial_matrix[start, end-processed_answer[0]] = \
          f1_score(start, end, processed_answer[0], processed_answer[1])
    data.append([processed_question, processed_answer, qid,
                 f1_partial_matrix, sentence_idx])

  return data, missed

class Data:
  def __init__(self, dictionary=None, immutable=False):
    self.dictionary = Dictionary(lowercase=False,
                                 remove_punctuation=False)
    if dictionary:
      self.dictionary = dictionary
    if immutable:
      self.dictionary.set_immutable()
    self.questions = {}
    self.questions_tokenized = {}
    self.questions_tokenized_words = {}
    self.answers = {}
    self.question_ner_tags = {}
    self.question_pos_tags = {}
    self.paragraphs = []
    self.tokenized_paras = []
    self.tokenized_para_words = []
    self.paras_pos_tags = []
    self.paras_ner_tags = []
    self.question_to_paragraph = {}
    self.data = []
    self.missed = 0

  def clear_aux_data(self):
    del self.questions
    del self.paragraphs
    del self.question_to_paragraph
    del self.questions_tokenized
    del self.questions_tokenized_words
    del self.answers
    del self.tokenized_paras
    del self.tokenized_para_words
    del self.data
    del self.question_ner_tags
    del self.question_pos_tags
    del self.paras_pos_tags
    del self.paras_ner_tags

  def dump_pickle(self, filename):
    with gzip.open(filename + ".gz", 'wb') as fout:
      pickle.dump(self, fout)
      fout.close()

  def read_from_pickle(self, filename):
    with gzip.open(filename + ".gz", 'rb') as fin:
      self = pickle.load(fin)
      fin.close()
      return self

  def get_ids(self, tokenized_text):
    return [ self.dictionary.add_or_get_index(word) \
               for word in tokenized_text ]

  def get_ids_immutable(self, tokenized_text):
    return [ self.dictionary.get_index(word) \
               for word in tokenized_text ]

  def add_paragraph(self, paragraph):
    para_text = paragraph['context']
    para_qas = paragraph['qas']

    for qa in para_qas:
      # Questions of length <= 2 words are ignored
      if len(qa['question'].split()) <= 2:
        continue
      self.question_to_paragraph[qa['id']] = len(self.paragraphs)
      self.questions[qa['id']] = qa['question']
      self.answers[qa['id']] = qa['answers']

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

    print "Tokenizing paragraphs (%d total)..." % len(self.paragraphs)
    _, self.tokenized_para_words, self.paras_pos_tags, self.paras_ner_tags = \
      zip(*Parallel(n_jobs=-1, verbose=2)(
        delayed(tokenize_and_tag)(None, para_text) for para_text in self.paragraphs))
    self.tokenized_para_words = list(self.tokenized_para_words)
    for tokenized_para_words in tqdm(self.tokenized_para_words):
      if tokenized_para_words is None:
        self.tokenized_paras.append(None)
        continue
      self.tokenized_paras.append(self.get_ids(tokenized_para_words))
    self.paras_pos_tags = list(self.paras_pos_tags)
    for sent_id, pos_tagged_para in tqdm(enumerate(self.paras_pos_tags)):
      if pos_tagged_para is None:
        continue
      assert len(pos_tagged_para) == len(self.tokenized_paras[sent_id]), str(sent_id)
      self.paras_pos_tags[sent_id] = \
        [ self.dictionary.add_or_get_postag(tag) for tag in pos_tagged_para ]
    self.paras_ner_tags = list(self.paras_ner_tags)
    for sent_id, ner_tagged_para in tqdm(enumerate(self.paras_ner_tags)):
      if ner_tagged_para is None:
        continue
      assert len(ner_tagged_para) == len(self.tokenized_paras[sent_id]), str(sent_id)
      self.paras_ner_tags[sent_id] = \
        [ self.dictionary.add_or_get_nertag(tag) for tag in ner_tagged_para ]
    print "Done!"

    print "Tokenizing questions (%d total)..." % len(self.questions)
    qids, self.questions_tokenized_words, self.question_pos_tags, self.question_ner_tags  = \
      zip(*Parallel(n_jobs=-1, verbose=10)(
        delayed(tokenize_and_tag)(qid, self.questions[qid]) for qid in self.questions))
    self.questions_tokenized_words = dict(zip(qids, self.questions_tokenized_words))
    self.question_pos_tags = dict(zip(qids, self.question_pos_tags))
    self.question_ner_tags = dict(zip(qids, self.question_ner_tags))
    for qid in tqdm(self.questions):
      if self.questions_tokenized_words[qid] is None:
        continue
      self.questions_tokenized[qid] = self.get_ids(self.questions_tokenized_words[qid])
    for qid in tqdm(self.question_pos_tags):
      if self.question_pos_tags[qid] is None:
        continue
      assert len(self.question_pos_tags[qid]) == len(self.questions_tokenized_words[qid]),\
             str(qid)
      self.question_pos_tags[qid] = \
        [ self.dictionary.add_or_get_postag(tag) for tag in self.question_pos_tags[qid] ]
    for qid in tqdm(self.question_ner_tags):
      if self.question_ner_tags[qid] is None:
        continue
      assert len(self.question_ner_tags[qid]) == len(self.question_pos_tags[qid]),\
             str(qid)
      self.question_ner_tags[qid] = \
        [ self.dictionary.add_or_get_nertag(tag) for tag in self.question_ner_tags[qid] ]
    print "Done!"

    to_process = (sum([ len(self.answers[qid]) for qid in self.questions ]))
    print "Creating data tuples for input (%d total)..." % to_process
    qtop = self.question_to_paragraph
    data, missed = \
      zip(*Parallel(n_jobs=-1, verbose=10, batch_size=10000)\
             (delayed(create_data)(qid, self.paragraphs[qtop[qid]],
                                   self.tokenized_paras[qtop[qid]],
                                   self.tokenized_para_words[qtop[qid]],
                                   self.questions_tokenized[qid], self.dictionary,
                                   self.questions[qid], self.answers[qid]) \
                for qid in self.questions_tokenized))
    self.data = [ item for sublist in data for item in sublist ]
    self.missed = sum(missed)
    print "Done!"

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

