import pdb
import argparse
import json
import numpy as np
import random
import sys
import time
import tensorflow as tf

from operator import itemgetter
from Input import Dictionary, Data, pad, read_data
from QA_Model import QA_Model

def init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_type', default='train')
  parser.add_argument('--train_json')
  parser.add_argument('--dev_json')
  parser.add_argument('--test_json')
  parser.add_argument('--dev_output_json')
  parser.add_argument('--train_pickle')
  parser.add_argument('--dev_pickle')
  parser.add_argument('--test_pickle')
  parser.add_argument('--predictions_output_json')
  parser.add_argument('--load_test', action='store_true')
  parser.add_argument('--dump_pickles', action='store_true')
  parser.add_argument('--num_incorrect_samples', type=int, default=5)
  parser.add_argument('--max_train_articles', type=int, default=-1)
  parser.add_argument('--max_dev_articles', type=int, default=-1)
  parser.add_argument('--embed_size', type=int, default=128)
  parser.add_argument('--hidden_size', type=int, default=256)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=0.02)
  parser.add_argument('--ckpt', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--model_dir', default='./')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--test_batch_size', type=int, default=512)
  parser.add_argument('--dropout_keep_value', type=float, default=0.5)
  parser.add_argument('--optimizer', default='SGD')
  parser.add_argument('--word2vec_path')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--nques_per_batch', type=int, default=10) 
  return parser


#------------- ---------------- Preprocess data -------------------------------#
def read_and_process_data(args):
  assert not (args.train_json == None and args.train_pickle == None)
  assert not (args.dev_json == None and args.dev_pickle == None)

  #----------------------- Read train, dev and test data ------------------------#
  train_data, dev_data, test_data = \
    read_data(args.train_json, args.train_pickle, args.dev_json, args.dev_pickle,
              args.test_json, args.test_pickle, args.num_incorrect_samples,
              args.max_train_articles, args.max_dev_articles, args.dump_pickles,
              args.dev_output_json, args.load_test)
  #------------------------------------------------------------------------------#

  train = train_data.data
  dev = dev_data.data
  test = test_data.data
  batch_size = args.batch_size
  test_batch_size = args.test_batch_size
  train_ques_to_para = train_data.question_to_paragraph
  dev_ques_to_para = dev_data.question_to_paragraph
  test_ques_to_para = test_data.question_to_paragraph
  train_tokenized_paras = train_data.tokenized_paras
  dev_tokenized_paras = dev_data.tokenized_paras
  test_tokenized_paras = test_data.tokenized_paras

  
  # Sort data by increasing question+answer length, for efficient batching.
  print "Sorting datasets by order of length."
  dev.sort(cmp=lambda x,y: len(x[0]) + len(x[1]) +
                             len(dev_tokenized_paras[dev_ques_to_para[x[3]]]) -\
                           (len(y[0]) + len(y[1]) +
                             len(dev_tokenized_paras[dev_ques_to_para[y[3]]])))
  test.sort(cmp=lambda x,y: len(x[0]) + len(x[1]) +
                              len(test_tokenized_paras[test_ques_to_para[x[3]]]) -\
                            (len(y[0]) + len(y[1]) +
                              len(test_tokenized_paras[test_ques_to_para[y[3]]])))
  print "Done."

  print "Computing stat variables."
  if args.debug:
    train = train[:32000]
    dev = dev[:32000]
    test = test[:32000]
  train_1_examples = sum([ example[2] for example in train ])
  dev_1_examples = sum([ example[2] for example in dev ])
  test_1_examples = sum([ example[2] for example in test ])

  # Prepare Train Batches 6 questions per batch 
  print "Preparing training batches."
  qid = train[0][3]
  count = 0
  prev_order = 0
  train_order = []
  for i in range(len(train)):
    if(train[i][3] != qid):
      count+=1
      qid = train[i][3]
    if(count == args.nques_per_batch):
      train_order.append((prev_order,i))
      prev_order = i
      count = 0
  print "Done."
      
  dev_order = [ i for i in range(0, len(dev), test_batch_size) ]
  test_order = [ i for i in range(0, len(test), test_batch_size) ]
  print "Done."

  return train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
         dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
         dev_tokenized_paras, test_tokenized_paras, train_1_examples,\
         dev_1_examples, test_1_examples, train_order, dev_order, test_order,\
         train_data, dev_data, test_data
#------------------------------------------------------------------------------#


#------------------------------ Create model ----------------------------------#
def build_model(args, nwords, index_to_word):
  config = { 'embed_size' : args.embed_size,
             'nwords' : nwords,
             'hidden_size' : args.hidden_size,
             'num_layers' : args.num_layers,
             'lr' : args.learning_rate,
             'cont': args.ckpt,
             'drop_emb': args.dropout_keep_value,
             'unequal_neg': True,
             'optimizer': args.optimizer,
             'incorrect_ratio': args.num_incorrect_samples,
             'word2vec_path': args.word2vec_path,
             'index_to_word': index_to_word }
  model = QA_Model(config)
  return model, config
#------------------------------------------------------------------------------#

def train_model(args):
  # Read and process data
  train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
  dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
  dev_tokenized_paras, test_tokenized_paras, train_1_examples, dev_1_examples,\
  test_1_examples, train_order, dev_order, test_order, train_data, dev_data,\
  test_data = read_and_process_data(args)
    
  # Build model
  model, config = build_model(args, train_data.dictionary.size(),
                              train_data.dictionary.index_to_word)

  #------------------------------ Train System ----------------------------------#
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True

  with tf.Session(config=tf_config) as sess:
    tf.global_variables_initializer().run()

    # Keep model parameters for all epochs
    saver = tf.train.Saver(max_to_keep=args.epochs)

    # Should we resume running from an existing checkpoint?
    last_done_epoch = config['cont']
    if last_done_epoch > 0:
      print "Continue from ckpt", last_done_epoch + 1
      saver.restore(sess, args.model_dir + 'model' + str(last_done_epoch) + '.ckpt')

    start_time = time.time()
    print "Starting training."

    for EPOCH in range(last_done_epoch+1, args.epochs):
      start_t = time.time()
      random.shuffle(train_order)
      train_loss_sum = 0.0
      for i, num in enumerate(train_order):
        print "\rTrain epoch %d, %.2f s - (Done %d of %d) " %\
              (EPOCH, (time.time()-start_t)*(len(train_order)-i-1)/(i+1), i+1,
               len(train_order)),

        # Create next batch by getting lengths and padding
        #if(i+1 == len(train_order)):
        #train_batch = train[train_order[i]:]
        train_batch = train[train_order[i][0]:train_order[i][1]]
        random.shuffle(train_batch)
        ans_lens_in = [ len(example[1]) for example in train_batch ]
        ques_lens_in = [ len(example[0]) for example in train_batch ]
        paras_in = [ train_tokenized_paras[train_ques_to_para[example[3]]] \
                       for example in train_batch ]
        paras_lens_in = [ len(para) for para in paras_in ]
        max_ans_len = max(ans_lens_in)
        max_ques_len = max(ques_lens_in)
        max_para_len = max(paras_lens_in)

        ans_in = [ pad(example[1], 0, max_ans_len) for example in train_batch ]
        ques_in = [ pad(example[0], 0, max_ques_len) for example in train_batch ]
        paras_in = [ pad(para, 0, max_para_len) for para in paras_in ]
        labels = [ example[2] for example in train_batch ]

        train_loss, predictions, _ =\
          sess.run([model.loss, model.predictions, model.optimizer],
                   feed_dict = { model.ans_input: ans_in,
                                 model.ans_lens: ans_lens_in,
                                 model.ques_input: ques_in,
                                 model.ques_lens: ques_lens_in,
                                 model.passage_input: paras_in,
                                 model.passage_lens: paras_lens_in,
                                 model.labels: labels, model.keep_prob:
                                 config['drop_emb'] })

        predictions = np.round(predictions)
        train_errors = np.abs(predictions-labels)
        train_error_sum = np.sum(train_errors)
        train_error0_sum = np.sum((1-np.array(labels)) * train_errors)
        train_error1_sum = np.sum(np.array(labels) * train_errors)
        train_1_examples_batch = sum([ example[2] for example in train_batch ])
        print ("Total error: %.2f%%, 1 errors: %.2f%%, " +\
              "0 errors: %.2f%%, ") %\
              (100 * float(train_error_sum)/len(train_batch),
               100 * float(train_error1_sum)/(1+train_1_examples_batch),
               100 * float(train_error0_sum)/(len(train_batch)-train_1_examples_batch)),
        sys.stdout.flush()
        train_loss_sum += train_loss
        print "Loss, %.5f" % (train_loss_sum/(i+1)),

        # Print train stats for epoch
        print "\nEpoch %d: Train Loss: %.4f (in time: %.2f s)" %\
              (EPOCH, train_loss_sum/len(train_order), (time.time() - start_t))

      # Run pass over dev data to compute stats
      dev_start_t = time.time()
      dev_loss_sum = 0.0
      dev_error_sum = 0
      dev_error0_sum = 0
      dev_error1_sum = 0
      for i, num in enumerate(dev_order):
        print "\rDev: %.2f s (Done %d of %d) " %\
              ((time.time()-dev_start_t)*(len(dev_order)-i-1)/(i+1), i+1,
              len(dev_order)),
        sys.stdout.flush()

        # Prepare dev bath by computing lengths and padding
        dev_batch = dev[num:num+test_batch_size]
        ans_lens_in = [ len(example[1]) for example in dev_batch ]
        ques_lens_in = [ len(example[0]) for example in dev_batch ]
        paras_in = [ dev_tokenized_paras[dev_ques_to_para[example[3]]] \
                       for example in dev_batch ]
        paras_lens_in = [ len(para) for para in paras_in ]
        max_ans_len = max(ans_lens_in)
        max_ques_len = max(ques_lens_in)
        max_para_len = max(paras_lens_in)

        ans_in = [ pad(example[1], 0, max_ans_len) for example in dev_batch ]
        ques_in = [ pad(example[0], 0, max_ques_len) for example in dev_batch ]
        paras_in = [ pad(para_in, 0, max_para_len) for para_in in paras_in ]
        labels = [ example[2] for example in dev_batch ]

        dev_loss, predictions =\
          sess.run([model.loss, model.predictions],
                   feed_dict = { model.ans_input: ans_in,
                                 model.ans_lens: ans_lens_in,
                                 model.ques_input: ques_in,
                                 model.ques_lens: ques_lens_in,
                                 model.passage_input: paras_in,
                                 model.passage_lens: paras_lens_in,
                                 model.labels: labels,
                                 model.keep_prob: 1.0 })

        dev_loss_sum += dev_loss
        print "[Average loss : %.5f]" % (dev_loss_sum/(i+1)),

        # Compute overall prediction-error, error for 0s, and error for 1s
        predictions = np.round(predictions)
        dev_errors = np.abs(predictions-labels)
        dev_error_sum += np.sum(dev_errors)
        dev_error0_sum += np.sum((1-np.array(labels)) * dev_errors)
        dev_error1_sum += np.sum(np.array(labels) * dev_errors)

      # Print dev stats for epoch
      print "\nDev Loss: %.4f (in time: %.2f s)" %\
            (dev_loss_sum/len(dev_order), (time.time() - dev_start_t))
      print ("Total error: %d/%d (%.2f%%), 1 errors: %d/%d (%.2f%%), " +\
             "0 errors: %d/%d (%.2f%%)") %\
             (dev_error_sum, len(dev), 100 * float(dev_error_sum)/len(dev),
              dev_error1_sum, dev_1_examples, 100 * float(dev_error1_sum)/dev_1_examples,
              dev_error0_sum, len(dev)-dev_1_examples,
              100 * float(dev_error0_sum)/(len(dev)-dev_1_examples))

      # Save model parameters from this epoch.
      save_path = saver.save(sess, args.model_dir + 'model' + str(EPOCH) + '.ckpt')
      print "Model saved."

#------------------------------------------------------------------------------#


#-------------------------------- Test model ----------------------------------#
def test_model(args):
  # Read and process data
  train, dev, test, batch_size, test_batch_size, train_ques_to_para,\
  dev_ques_to_para, test_ques_to_para, train_tokenized_paras,\
  dev_tokenized_paras, test_tokenized_paras, train_1_examples, dev_1_examples,\
  test_1_examples, train_order, dev_order, test_order, train_data, dev_data,\
  test_data = read_and_process_data(args)

  # Build model
  print "Building model."
  model, config = build_model(args, train_data.dictionary.size(),
                              train_data.dictionary.index_to_word)
  print "Done."

  #------------------------- Reload and test model ----------------------------#
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True

  with tf.Session(config=tf_config) as sess:
    print "Initializing variables."
    tf.global_variables_initializer().run()
    print "Done."
    assert not args.ckpt == 0
    assert not args.predictions_output_json is None

    saver = tf.train.Saver(max_to_keep=args.epochs)
    print "Loading model from checkpoint."
    saver.restore(sess, args.model_dir + 'model' + str(args.ckpt) + '.ckpt')
    print "Done."

    # Run pass over test data to compute stats
    test_start_t = time.time()
    test_loss_sum = 0.0
    test_error_sum = 0
    test_error0_sum = 0
    test_error1_sum = 0
    all_predictions = {}
    for i, num in enumerate(test_order):
      print "\rTest: %.2f s (Done %d of %d) " %\
        ((time.time()-test_start_t)*(len(test_order)-i-1)/(i+1), i+1, len(test_order)),
      sys.stdout.flush()

      # Prepare test batch by computing lengths and padding
      test_batch = test[num:num+test_batch_size]
      ans_lens_in = [ len(example[1]) for example in test_batch ]
      ques_lens_in = [ len(example[0]) for example in test_batch ]
      paras_in = [ test_tokenized_paras[test_ques_to_para[example[3]]] \
                     for example in test_batch ]
      paras_lens_in = [ len(para) for para in paras_in ]
      max_ans_len = max(ans_lens_in)
      max_ques_len = max(ques_lens_in)
      max_para_len = max(paras_lens_in)

      ans_in = [ pad(example[1], 0, max_ans_len) for example in test_batch ]
      ques_in = [ pad(example[0], 0, max_ques_len) for example in test_batch ]
      paras_in = [ pad(para_in, 0, max_para_len) for para_in in paras_in ]
      labels = [ example[2] for example in test_batch ]

      # Add all batch qids to predictions dict, if they don't already exist
      qids = [ example[3] for example in test_batch ]
      answers = [ " ".join([ test_data.dictionary.get_word(idx) for idx in example[1] ]) \
                    for example in test_batch ]

      for qid in qids:
        if not qid in all_predictions:
          all_predictions[qid] = []

      test_loss, predictions =\
        sess.run([model.loss, model.predictions],
                 feed_dict = { model.ans_input: ans_in,
                               model.ans_lens: ans_lens_in,
                               model.ques_input: ques_in,
                               model.ques_lens: ques_lens_in,
                               model.passage_input: paras_in,
                               model.passage_lens: paras_lens_in,
                               model.labels: labels,
                               model.keep_prob: 1.0 })

      test_loss_sum += test_loss
      print "[Average loss : %.5f]" % (test_loss_sum/(i+1)),

      for qid, answer, prob in zip(qids, answers, predictions):
        all_predictions[qid].append([answer, prob])

      # Compute overall prediction-error, error for 0s, and error for 1s
      predictions = np.round(predictions)
      test_errors = np.abs(predictions-labels)
      test_error_sum += np.sum(test_errors)
      test_error0_sum += np.sum((1-np.array(labels)) * test_errors)
      test_error1_sum += np.sum(np.array(labels) * test_errors)

    # Print test stats for epoch
    print "\nTest Loss: %.4f (in time: %.2f s)" %\
    (test_loss_sum/len(test_order), (time.time() - test_start_t))
    print ("Total error: %d/%d (%.2f%%), 1 errors: %d/%d (%.2f%%), " +\
           "0 errors: %d/%d (%.2f%%)") %\
           (test_error_sum, len(test), 100 * float(test_error_sum)/len(test),
            test_error1_sum, test_1_examples, 100 * float(test_error1_sum)/test_1_examples,
            test_error0_sum, len(test)-test_1_examples,
            100 * float(test_error0_sum)/(len(test)-test_1_examples))

    # Select the best answer for each question (highest probability)
    print "Getting best answers."
    for qid in all_predictions:
      all_predictions[qid] = max(all_predictions[qid], key=itemgetter(1))[0]
    print "Done."

    # Dump the results json in the required format
    print "Dumping prediction results."
    with open(args.predictions_output_json, "w") as predictions_out:
      json.dump(all_predictions, predictions_out)
      predictions_out.close()
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

