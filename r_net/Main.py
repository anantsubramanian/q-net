import argparse
import json
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn

from operator import itemgetter
from torch.autograd import Variable
from torch.optim import SGD, Adamax, Adadelta
from Input import Dictionary, Data, pad, read_data
from rNet import rNet

def init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_type', default='train')
  parser.add_argument('--train_json')
  parser.add_argument('--dev_json')
  parser.add_argument('--train_pickle')
  parser.add_argument('--dev_pickle')
  parser.add_argument('--predictions_output_json')
  parser.add_argument('--dump_pickles', action='store_true')
  parser.add_argument('--max_train_articles', type=int, default=-1)
  parser.add_argument('--max_dev_articles', type=int, default=-1)
  parser.add_argument('--embed_size', type=int, default=300)
  parser.add_argument('--hidden_size', type=int, default=75)
  parser.add_argument('--learning_rate', type=float, default=0.005)
  parser.add_argument('--glove_path', default='../../data/glove/glove.840B.300d.txt')
  parser.add_argument('--disable_glove', action='store_true')
  parser.add_argument('--ckpt', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=25)
  parser.add_argument('--model_dir', default='./')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--test_batch_size', type=int, default=32)
  parser.add_argument('--optimizer', default='Adamax') # 'SGD' or 'Adamax' or Adadelta
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--dropout', type=float, default=0.2)
  parser.add_argument('--decay', type=float, default=0.95)
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--max_answer_span', type=int, default=15)
  return parser


#------------- ---------------- Preprocess data -------------------------------#
def read_and_process_data(args):
  assert not (args.train_json == None and args.train_pickle == None)
  assert not (args.dev_json == None and args.dev_pickle == None)

  #----------------------- Read train, dev and test data ------------------------#
  train_data, dev_data = \
    read_data(args.train_json, args.train_pickle, args.dev_json, args.dev_pickle,
              args.max_train_articles, args.max_dev_articles, args.dump_pickles)
  #------------------------------------------------------------------------------#

  # Our dev is also test...
  test_data = dev_data

  train = train_data.data
  dev = dev_data.data
  test = dev_data.data
  batch_size = args.batch_size
  test_batch_size = args.test_batch_size
  train_ques_to_para = train_data.question_to_paragraph
  dev_ques_to_para = dev_data.question_to_paragraph
  test_ques_to_para = dev_data.question_to_paragraph
  train_tokenized_paras = train_data.tokenized_paras
  dev_tokenized_paras = dev_data.tokenized_paras
  test_tokenized_paras = dev_data.tokenized_paras
  train_tokenized_paras_chars = train_data.tokenized_paras_chars
  dev_tokenized_paras_chars = dev_data.tokenized_paras_chars
  test_tokenized_paras_chars = test_data.tokenized_paras_chars

  # Sort data by increasing question+answer length, for efficient batching.
  # Data format = (tokenized_question, tokenized_answer, question_id).
  print "Sorting datasets in decreasing order of (para + question) lengths."
  train.sort(cmp=lambda x,y:\
      len(x[0]) + len(train_tokenized_paras[train_ques_to_para[x[2]]]) -\
      (len(y[0]) + len(train_tokenized_paras[train_ques_to_para[y[2]]])),
      reverse=True)
  dev.sort(cmp=lambda x,y:\
      len(x[0]) + len(dev_tokenized_paras[dev_ques_to_para[x[2]]]) -\
      (len(y[0]) + len(dev_tokenized_paras[dev_ques_to_para[y[2]]])),
      reverse=True)
  test.sort(cmp=lambda x,y:\
      len(x[0]) + len(test_tokenized_paras[test_ques_to_para[x[2]]]) -\
      (len(y[0]) + len(test_tokenized_paras[test_ques_to_para[y[2]]])),
      reverse=True)
  print "Done."

  # Debug flag reduces size of input data, for testing purposes.
  if args.debug:
    train = train[:320]
    dev = dev[:320]
    test = test[:320]

  train_order = [ i for i in range(0, len(train), batch_size) ]
  dev_order = [ i for i in range(0, len(dev), test_batch_size) ]
  test_order = [ i for i in range(0, len(test), test_batch_size) ]
  print "Done."

  return train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
         dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
         dev_tokenized_paras, test_tokenized_paras, train_order, dev_order,\
         test_order, train_data, dev_data, test_data, train_tokenized_paras_chars,\
         dev_tokenized_paras_chars, test_tokenized_paras_chars
#------------------------------------------------------------------------------#


#------------------------------ Create model ----------------------------------#
def build_model(args, vocab_size, index_to_word, word_to_index, char_to_index,
                index_to_char):
  config = { 'embed_size' : args.embed_size,
             'vocab_size' : vocab_size,
             'char_vocab_size' : len(index_to_char),
             'hidden_size' : args.hidden_size,
             'lr' : args.learning_rate,
             'glove_path' : args.glove_path,
             'use_glove' : not args.disable_glove,
             'ckpt': args.ckpt,
             'optimizer': args.optimizer,
             'index_to_word': index_to_word,
             'word_to_index': word_to_index,
             'char_to_index': char_to_index,
             'index_to_char': index_to_char,
             'dropout' : args.dropout,
             'cuda': args.cuda,
             'decay': args.decay }
  print "Building model."
  model = rNet(config)
  print "Done!"
  sys.stdout.flush()

  print "%d OOV words." % model.oov_count
  print "OOV Words:",
  print model.oov_list[:10]

  if not args.disable_glove:
    print "Embedding dim:", model.embedding.shape

  if args.cuda:
    model = model.cuda()

  return model, config
#------------------------------------------------------------------------------#

def get_minibatch_input(minibatch, tokenized_paras, tokenized_paras_chars,
                        ques_to_para):
  # Variable length question, answer and paragraph sequences for batch.
  ques_lens_in = [ len(example[0]) for example in minibatch ]
  paras_in = [ tokenized_paras[ques_to_para[example[2]]] \
                 for example in minibatch ]
  paras_chars_in = [ tokenized_paras_chars[ques_to_para[example[2]]] \
                       for example in minibatch ]
  paras_chars_in_b = [ [ c[::-1] for c in f ] for f in paras_chars_in ]
  paras_lens_in = [ len(para) for para in paras_in ]

  max_ques_len = max(ques_lens_in)
  max_para_len = max(paras_lens_in)

  ques_chars_forward = [ example[3] for example in minibatch ]
  ques_chars_backward = [ [ c[::-1] for c in f ] for f in ques_chars_forward ]
  ques_chars_lens_in = [ [ len(x) for x in ques_word_chars ] \
                           for ques_word_chars in ques_chars_forward ]
  paras_chars_lens_in = [ [ len(x) for x in para_word_chars ] \
                           for para_word_chars in paras_chars_in ]

  max_ques_wordlen = max([ max(x) for x in ques_chars_lens_in ])
  max_paras_wordlen = max([ max(x) for x in paras_chars_lens_in ])

  ques_chars_lens_in = [ pad(x, 1, max_ques_len) for x in ques_chars_lens_in ]
  paras_chars_lens_in = [ pad(x, 1, max_para_len) for x in paras_chars_lens_in ]

  # Question forward character LSTM input.
  ques_chars_forward_in = []
  zero_padded_word = [0] * max_ques_wordlen
  for ques in ques_chars_forward:
    ques_words = []
    for word in ques:
      ques_words.append(pad(word, 0, max_ques_wordlen))
    ques_chars_forward_in.append(pad(ques_words, zero_padded_word, max_ques_len))
  ques_chars_forward_in = np.array(ques_chars_forward_in)

  # Question backward character LSTM input.
  ques_chars_backward_in = []
  for ques in ques_chars_backward:
    ques_words = []
    for word in ques:
      ques_words.append(pad(word, 0, max_ques_wordlen))
    ques_chars_backward_in.append(pad(ques_words, zero_padded_word, max_ques_len))
  ques_chars_backward_in = np.array(ques_chars_backward_in)

  # Passage forward character LSTM input.
  paras_chars_forward_in = []
  zero_padded_word = [0] * max_paras_wordlen
  for para in paras_chars_in:
    para_words = []
    for word in para:
      para_words.append(pad(word, 0, max_paras_wordlen))
    paras_chars_forward_in.append(pad(para_words, zero_padded_word, max_para_len))
  paras_chars_forward_in = np.array(paras_chars_forward_in)

  # Passage backward character LSTM input.
  paras_chars_backward_in = []
  for para in paras_chars_in_b:
    para_words = []
    for word in para:
      para_words.append(pad(word, 0, max_paras_wordlen))
    paras_chars_backward_in.append(pad(para_words, zero_padded_word, max_para_len))
  paras_chars_backward_in = np.array(paras_chars_backward_in)

  # ans_in.shape = (2, batch)
  ans_in = np.array([ example[1] for example in minibatch ]).T

  # Fixed-length (padded) input sequences with shape=(seq_len, batch).
  ques_in_f = np.array([ pad(example[0], 0, max_ques_len)\
                         for example in minibatch ]).T
  paras_in_f = np.array([ pad(para, 0, max_para_len) for para in paras_in ]).T
  ques_in_b = np.array([ pad(example[0][::-1], 0, max_ques_len)\
                           for example in minibatch ]).T
  paras_in_b = np.array([ pad(para[::-1], 0, max_para_len) for para in paras_in ]).T

  passage_input_f = paras_in_f
  passage_input_b = paras_in_b
  question_input_f = ques_in_f
  question_input_b = ques_in_b
  passage_input_lens = paras_lens_in
  question_input_lens = ques_lens_in
  passage_input_chars_f = np.transpose(paras_chars_forward_in, (1, 0, 2))
  passage_input_chars_b = np.transpose(paras_chars_backward_in, (1, 0, 2))
  question_input_chars_f = np.transpose(ques_chars_forward_in, (1, 0, 2))
  question_input_chars_b = np.transpose(ques_chars_backward_in, (1, 0, 2))
  passage_input_chars_lens = np.transpose(np.array(paras_chars_lens_in))
  question_input_chars_lens = np.transpose(np.array(ques_chars_lens_in))
  answer_input = ans_in

  return passage_input_f, passage_input_b, question_input_f, question_input_b,\
         passage_input_lens, question_input_lens, passage_input_chars_f,\
         passage_input_chars_b, question_input_chars_f, question_input_chars_b,\
         passage_input_chars_lens, question_input_chars_lens, answer_input


def train_model(args):
  # Read and process data
  train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
  dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
  dev_tokenized_paras, test_tokenized_paras, train_order, dev_order, test_order,\
  train_data, dev_data, test_data, train_tokenized_paras_chars,\
  dev_tokenized_paras_chars, test_tokenized_paras_chars = read_and_process_data(args)

  # Build model
  model, config = build_model(args, train_data.dictionary.size(),
                              train_data.dictionary.index_to_word,
                              train_data.dictionary.word_to_index,
                              train_data.dictionary.char_to_index,
                              train_data.dictionary.index_to_char)

  if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)

  #------------------------------ Train System ----------------------------------#
  # Should we resume running from an existing checkpoint?
  last_done_epoch = config['ckpt']
  if last_done_epoch > 0:
    model = model.load(args.model_dir, last_done_epoch)
    print "Loaded model."
    if not args.disable_glove:
      print "Embedding shape:", model.embedding.shape

  start_time = time.time()
  print "Starting training."

  # Decide which optimizer to use.
  if args.optimizer == "SGD":
    print "Using SGD optimizer."
    optimizer = SGD(model.parameters(), lr = args.learning_rate)
  elif args.optimizer == "Adamax":
    print "Using Adamax optimizer."
    optimizer = Adamax(model.parameters(), lr= args.learning_rate)
  elif args.optimizer == "Adadelta":
    print "Using Adadelta optimizer."
    optimizer = Adadelta(model.parameters(), lr=args.learning_rate, rho=0.95)
  else:
    assert False, "Unrecognized optimizer."

  if last_done_epoch > 0:
    if os.path.exists(args.model_dir + "/optim_%d.pt" % last_done_epoch):
      optimizer = torch.load(args.model_dir + "/optim_%d.pt" % last_done_epoch)
    else:
      print "Optimizer saved state not found. Not loading optimizer."

  # Model summary.
  print(model)

  for EPOCH in range(last_done_epoch+1, args.epochs):
    start_t = time.time()
    train_loss_sum = 0.0
    model.train()
    for i, num in enumerate(train_order):
      print "\rTrain epoch %d, %.2f s - (Done %d of %d)" %\
            (EPOCH, (time.time()-start_t)*(len(train_order)-i-1)/(i+1), i+1,
             len(train_order)),

      # Create next batch by getting lengths and padding
      train_batch = train[num:num+batch_size]

      passage_input_f, passage_input_b, question_input_f, question_input_b,\
      passage_input_lens, question_input_lens, passage_input_chars_f,\
      passage_input_chars_b, question_input_chars_f, question_input_chars_b,\
      passage_input_chars_lens, question_input_chars_lens, answer_input =\
        get_minibatch_input(train_batch, train_tokenized_paras,
                            train_tokenized_paras_chars, train_ques_to_para)

      # Zero previous gradient.
      model.zero_grad()
      model((passage_input_chars_f, passage_input_chars_lens),\
            (passage_input_chars_b, passage_input_chars_lens),\
            (question_input_chars_f, question_input_chars_lens),\
            (question_input_chars_b, question_input_chars_lens),\
            (passage_input_f, passage_input_lens),\
            (passage_input_b, passage_input_lens),\
            (question_input_f, question_input_lens),\
            (question_input_b, question_input_lens),\
            answer_input)

      model.loss.backward()
      optimizer.step()
      train_loss_sum += model.loss.data[0]
      model.free_memory()

      print "Loss: %.5f (in time %.2fs)" % \
            (train_loss_sum/(i+1), time.time() - start_t),
      sys.stdout.flush()

    print "\nLoss: %.5f (in time %.2fs)" % \
          (train_loss_sum/len(train_order), time.time() - start_t)

    # End of epoch.
    random.shuffle(train_order)
    model.zero_grad()
    model.save(args.model_dir, EPOCH)

    # Updating LR for optimizer
    for param in optimizer.param_groups:
      param['lr'] *= config['decay']

    torch.save(optimizer, args.model_dir + "/optim_%d.pt" % EPOCH)

    # Run pass over dev data.
    dev_start_t = time.time()
    dev_loss_sum = 0.0
    all_predictions = {}
    print "\nRunning on Dev."

    model.eval()
    for i, num in enumerate(dev_order):
      print "\rDev: %.2f s (Done %d of %d)" %\
            ((time.time()-dev_start_t)*(len(dev_order)-i-1)/(i+1), i+1,
            len(dev_order)),

      dev_batch = dev[num:num+test_batch_size]

      passage_input_f, passage_input_b, question_input_f, question_input_b,\
      passage_input_lens, question_input_lens, passage_input_chars_f,\
      passage_input_chars_b, question_input_chars_f, question_input_chars_b,\
      passage_input_chars_lens, question_input_chars_lens, answer_input =\
        get_minibatch_input(dev_batch, dev_tokenized_paras,
                            dev_tokenized_paras_chars, dev_ques_to_para)

      # distributions[{0,1}].shape = (batch, max_passage_len)
      distributions = \
        model((passage_input_chars_f, passage_input_chars_lens),\
              (passage_input_chars_b, passage_input_chars_lens),\
              (question_input_chars_f, question_input_chars_lens),\
              (question_input_chars_b, question_input_chars_lens),\
              (passage_input_f, passage_input_lens),\
              (passage_input_b, passage_input_lens),\
              (question_input_f, question_input_lens),\
              (question_input_b, question_input_lens),\
              answer_input)
      distributions[0] = distributions[0].data.cpu().numpy()
      distributions[1] = distributions[1].data.cpu().numpy()

      # Add all batch qids to predictions dict, if they don't already exist.
      qids = [ example[2] for example in dev_batch ]
      for qid in qids:
        if not qid in all_predictions:
          all_predictions[qid] = []

      best_idxs = []
      for idx in range(len(dev_batch)):
        best_prob = -1
        best = [0, 0]
        max_end = passage_input_lens[idx]
        for j, start_prob in enumerate(distributions[0][idx][:max_end]):
          cur_end_idx = min(j + args.max_answer_span, max_end)
          end_idx = np.argmax(distributions[1][idx][j:cur_end_idx])
          prob = distributions[1][idx][j+end_idx] * start_prob
          if prob > best_prob:
            best_prob = prob
            best = [j, j+end_idx]
        best_idxs.append(best)

      tokenized_paras = dev_data.tokenized_paras
      answers = [ tokenized_paras[dev_ques_to_para[qids[idx]]][start:end+1] \
                    for idx, (start, end) in enumerate(best_idxs) ]
      answers = [ " ".join([ dev_data.dictionary.get_word(idx) for idx in ans ]) \
                    for ans in answers ]

      for qid, answer in zip(qids, answers):
        all_predictions[qid] = answer

      dev_loss_sum += model.loss.data[0]
      model.free_memory()
      print "[Average loss : %.5f]" % (dev_loss_sum/(i+1)),
      sys.stdout.flush()

    # Print dev stats for epoch
    print "\nDev Loss: %.4f (in time: %.2f s)" %\
          (dev_loss_sum/len(dev_order), (time.time() - dev_start_t))

    # Dump the results json in the required format
    print "Dumping prediction results."
    json.dump(
      all_predictions,
      open(args.model_dir + "/dev_predictions_" + str(EPOCH) + ".json", "w"))
    print "Done."
#------------------------------------------------------------------------------#


#-------------------------------- Test model ----------------------------------#
def test_model(args):
  # Read and process data
  train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
  dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
  dev_tokenized_paras, test_tokenized_paras, train_order, dev_order, test_order,\
  train_data, dev_data, test_data, train_tokenized_paras_chars,\
  dev_tokenized_paras_chars, test_tokenized_paras_chars = read_and_process_data(args)

  # Build model
  model, config = build_model(args, train_data.dictionary.size(),
                              train_data.dictionary.index_to_word,
                              train_data.dictionary.word_to_index,
                              train_data.dictionary.char_to_index,
                              train_data.dictionary.index_to_char)
  print(model)

  #------------------------- Reload and test model ----------------------------#
  last_done_epoch = config['ckpt']
  model = model.load(args.model_dir, last_done_epoch)
  print "Loaded model."
  if not args.disable_glove:
    print "Embedding shape:", model.embedding.shape

  test_start_t = time.time()
  test_loss_sum = 0.0
  all_predictions = {}
  model.eval()

  for i, num in enumerate(test_order):
    print "\rTest: %.2f s (Done %d of %d) " %\
          ((time.time()-test_start_t)*(len(test_order)-i-1)/(i+1), i+1,
          len(test_order)),

    test_batch = test[num:num+test_batch_size]

    passage_input_f, passage_input_b, question_input_f, question_input_b,\
    passage_input_lens, question_input_lens, passage_input_chars_f,\
    passage_input_chars_b, question_input_chars_f, question_input_chars_b,\
    passage_input_chars_lens, question_input_chars_lens, answer_input =\
      get_minibatch_input(test_batch, test_tokenized_paras,
                          test_tokenized_paras_chars, test_ques_to_para)

    # distributions[{0,1}].shape = (batch, max_passage_len)
    distributions = \
      model((passage_input_chars_f, passage_input_chars_lens),\
            (passage_input_chars_b, passage_input_chars_lens),\
            (question_input_chars_f, question_input_chars_lens),\
            (question_input_chars_b, question_input_chars_lens),\
            (passage_input_f, passage_input_lens),\
            (passage_input_b, passage_input_lens),\
            (question_input_f, question_input_lens),\
            (question_input_b, question_input_lens),\
            answer_input)
    distributions[0] = distributions[0].data.cpu().numpy()
    distributions[1] = distributions[1].data.cpu().numpy()

    # Add all batch qids to predictions dict, if they don't already exist.
    qids = [ example[2] for example in test_batch ]
    for qid in qids:
      if not qid in all_predictions:
        all_predictions[qid] = []

    best_idxs = []
    for idx in range(len(test_batch)):
      best_prob = -1
      best = [0, 0]
      max_end = passage_input_lens[idx]
      for j, start_prob in enumerate(distributions[0][idx][:max_end]):
        cur_end_idx = min(j + args.max_answer_span, max_end)
        end_idx = np.argmax(distributions[1][idx][j:cur_end_idx])
        prob = distributions[1][idx][j+end_idx] * start_prob
        if prob > best_prob:
          best_prob = prob
          best = [j, j+end_idx]
      best_idxs.append(best)

    tokenized_paras = test_data.tokenized_paras
    answers = [ tokenized_paras[test_ques_to_para[qids[idx]]][start:end+1] \
                  for idx, (start, end) in enumerate(best_idxs) ]
    answers = [ " ".join([ test_data.dictionary.get_word(idx) for idx in ans ]) \
                  for ans in answers ]

    for qid, answer in zip(qids, answers):
      all_predictions[qid] = answer

    test_loss_sum += model.loss.data[0]
    model.free_memory()
    print "[Average loss : %.5f]" % (test_loss_sum/(i+1)),
    sys.stdout.flush()

  # Print stats
  print "\nTest Loss: %.4f (in time: %.2f s)" %\
        (test_loss_sum/len(test_order), (time.time() - test_start_t))

  # Dump the results json in the required format
  print "Dumping prediction results."
  json.dump(all_predictions, open(args.predictions_output_json, "w"))
  print "Done."
#------------------------------------------------------------------------------#

if __name__ == "__main__":
  args = init_parser().parse_args()
  if args.run_type == "train":
    train_model(args)
  elif args.run_type == "test":
    test_model(args)
  else:
    print "Invalid run type:", args.run_type

