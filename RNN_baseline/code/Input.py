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
    self.tokenized_paras = []
    self.question_to_paragraph = {}
    self.data = []
    self.num_incorrect_candidates = num_incorrect_candidates

  def clear_aux_data(self):
    del self.questions
    del self.paragraphs
    del self.question_to_paragraph
    del self.tokenized_paras
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

  def get_str(self, tokens):
    return ",".join([ "<" + str(token) + ">" for token in tokens ])

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
        correct_answers.add(self.get_str(answer))
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
      joined_para = []
      map(joined_para.extend, tokenized_para)
      self.tokenized_paras.append(joined_para)
      
      sentences = len(tokenized_para)
      if sentences == 0:
        print "Found no sentences for para:", para_text
        continue
      sent_lens = [ len(x) for x in tokenized_para ]

      # If the number of incorrect candidates to sample is set, use sampling.
      # Otherwise, generate ~L(L-1)/2 candidate answers.
      # While generating num_incorrect_candidates, we give up if we can't generate
      # a new candidate in 200 tries.
      if self.num_incorrect_candidates:
        candidate_answers = []
        tries = 0
        prev = len(candidate_answers)
        while len(candidate_answers) < self.num_incorrect_candidates:
          if len(candidate_answers) == prev:
            tries += 1
            if tries == 200:
              break
          prev = len(candidate_answers)
          sent_index = numpy.random.randint(0, sentences)
          start_index = numpy.random.randint(0, sent_lens[sent_index])
          end_index = numpy.random.randint(start_index+1, sent_lens[sent_index] + 1)
          candidate = tokenized_para[sent_index][start_index:end_index]
          candidate_str = self.get_str(candidate)
          if not candidate_str in seen_answers and not candidate_str in correct_answers:
            is_superset = False
            for correct_answer in correct_answers:
              # If candidate answer is a super set of a correct answer, we ignore it.
              if correct_answer in candidate_str:
                is_superset = True
                break
            if is_superset:
              continue
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
              candidate_str = self.get_str(candidate_answer)
              if candidate_str in correct_answers or candidate_str in seen:
                continue
              is_superset = False
              for correct_answer in correct_answers:
                if correct_answer in candidate_str:
                  is_superset = True
                  break
              if is_superset:
                continue
              self.data.append([processed_question, candidate_answer, 0.0, qa['id']])
              seen.add(candidate_str)

    # Store paragraph text
    self.paragraphs.append(para_text)


  def read_from_file(self, filename, max_articles, return_test=False,
                     test_split=10):
    dev_data = {}
    if return_test:
      test_data = {}
    with open(filename, 'r') as input_file:
      data = json.load(input_file)
      dev_data['version'] = data['version']
      dev_data['data'] = []
      if return_test:
        test_data['version'] = data['version']
        test_data['data'] = []


      data = data['data']
      # Read each input article
      for article_index, article in enumerate(data):
        if article_index == max_articles:
          break
        if return_test and article_index >= test_split:
          test_data['data'].append(article)
          continue
        dev_data['data'].append(article)
        # Read each para for each article
        for para_index, paragraph in enumerate(article['paragraphs']):
          self.add_paragraph(paragraph)
          print "\r%d Articles, %d Paragraphs processed." \
                  % (article_index+1, para_index+1),
          sys.stdout.flush()
      print ""

    if return_test:
      return dev_data, test_data
    return dev_data, None

def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r

def read_data(train_json, train_pickle, dev_json, dev_pickle, test_json, test_pickle,
              num_incorrect_samples, max_train_articles, max_dev_articles,
              dump_pickles, dev_output_json, load_test=False):
  train_data = Data(num_incorrect_candidates=num_incorrect_samples)
  print "Reading training data."
  if train_json:
    train_data.read_from_file(train_json, max_train_articles)
    if dump_pickles:
      assert not train_pickle == None
      train_data.dump_pickle(train_pickle)
  else:
    train_data = train_data.read_from_pickle(train_pickle)

  print "Done."
  print "Vocab size is:", train_data.dictionary.size()
  print "Sample words:", train_data.dictionary.index_to_word[:10]

  dev_data = Data(train_data.dictionary)
  test_data = Data(train_data.dictionary)
  if dev_json:
    assert test_json is not None
    assert dev_output_json is not None
    print "Reading dev data."
    dev_json_data, test_json_data = \
      dev_data.read_from_file(dev_json,max_dev_articles, return_test=True)
    with open(test_json, "w") as test_out:
      json.dump(test_json_data, test_out)
      test_out.close()
    with open(dev_output_json, "w") as dev_out:
      json.dump(dev_json_data, dev_out)
      dev_out.close()
    print "Done."

    if load_test:
      print "Reading test data."
      test_data = Data(train_data.dictionary)
      test_data.read_from_file(test_json, max_dev_articles)
      print "Done."

    if dump_pickles:
      assert not dev_pickle == None and not test_pickle == None
      print "Dumping pickles."
      dev_data.dump_pickle(dev_pickle)
      test_data.dump_pickle(test_pickle)
      print "Done."
  else:
    print "Reading dev data."
    dev_data = dev_data.read_from_pickle(dev_pickle)
    print "Done."
    if load_test:
      print "Reading test data."
      test_data = test_data.read_from_pickle(test_pickle)
      print "Done."

  print "Finished reading all required data."
  return train_data, dev_data, test_data

