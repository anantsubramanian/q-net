import argparse
import numpy as np
import random
import sys
import time
import tensorflow as tf

from Input import Dictionary, Data, pad, read_data
from QA_Model import QA_Model

def init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_json')
  parser.add_argument('--dev_json')
  parser.add_argument('--train_pickle')
  parser.add_argument('--dev_pickle')
  parser.add_argument('--test_pickle')
  parser.add_argument('--dump_pickles', action='store_true')
  parser.add_argument('--num_incorrect_samples', type=int, default=5)
  parser.add_argument('--max_train_articles', type=int, default=-1)
  parser.add_argument('--max_dev_articles', type=int, default=-1)
  parser.add_argument('--embed_size', type=int, default=300)
  parser.add_argument('--hidden_size', type=int, default=200)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=0.01)
  parser.add_argument('--ckpt', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--model_dir', default='./')
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--test_batch_size', type=int, default=1024)
  parser.add_argument('--dropout_keep_value', type=float, default=1.0)
  parser.add_argument('--optimizer', default='Momentum')
  parser.add_argument('--word2vec_path')
  return parser

args = init_parser().parse_args()
assert not (args.train_json == None and args.train_pickle == None)
assert not (args.dev_json == None and args.dev_pickle == None)

#----------------------- Read train, dev and test data ------------------------#
train_data, dev_data, test_data = \
  read_data(args.train_json, args.train_pickle, args.dev_json, args.dev_pickle,
            args.test_pickle, args.num_incorrect_samples, args.max_train_articles,
            args.max_dev_articles, args.dump_pickles)
#------------------------------------------------------------------------------#

#------------- ---------------- Preprocess data -------------------------------#
train = train_data.data
dev = dev_data.data
test = test_data.data
batch_size = args.batch_size
test_batch_size = args.test_batch_size

# Sort data by increasing question+answer length, for efficient batching.
train.sort(cmp=lambda x,y: len(x[0]) + len(x[1]) - (len(y[0]) + len(y[1])))
dev.sort(cmp=lambda x,y: len(x[0]) + len(x[1]) - (len(y[0]) + len(y[1])))
train_1_examples = sum([ example[2] for example in train ])
dev_1_examples = sum([ example[2] for example in dev ])
test_1_examples = sum([ example[2] for example in test ])

train_order = [ i for i in range(0, len(train), batch_size) ]
dev_order = [ i for i in range(0, len(dev), test_batch_size) ]
test_order = [ i for i in range(0, len(test), test_batch_size) ]
#------------------------------------------------------------------------------#

#------------------------------ Create model ----------------------------------#
config = { 'embed_size' : args.embed_size,
           'nwords' : train_data.dictionary.size(),
           'hidden_size' : args.hidden_size,
           'num_layers' : args.num_layers,
           'lr' : args.learning_rate,
           'cont': args.ckpt,
           'drop_emb': args.dropout_keep_value,
           'unequal_neg': True,
           'optimizer': args.optimizer,
           'incorrect_ratio': args.num_incorrect_samples,
           'word2vec_path': args.word2vec_path,
           'index_to_word': train_data.dictionary.index_to_word }
model = QA_Model(config)
#------------------------------------------------------------------------------#

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

  for EPOCH in range(last_done_epoch + 1, args.epochs):
      start_t = time.time()
      random.shuffle(train_order)
      train_loss_sum = 0.0
      for i, num in enumerate(train_order):
          print "\r[Train] - [%.2f s left] - [%d of %d batches] " %\
            ((time.time()-start_t)*(len(train_order)-i-1)/(i+1), i+1, len(train_order)),
          sys.stdout.flush()

          # Create next batch by getting lengths and padding
          train_batch = train[num:num+batch_size]
          ans_lens_in = [ len(example[1]) for example in train_batch ]
          ques_lens_in = [ len(example[0]) for example in train_batch ]
          max_ans_len = max(ans_lens_in)
          max_ques_len = max(ques_lens_in)

          ans_in = [ pad(example[1], 0, max_ans_len) for example in train_batch ]
          ques_in = [ pad(example[0], 0, max_ques_len) for example in train_batch ]
          labels = [ example[2] for example in train_batch ]

          train_loss, _ =\
            sess.run([model.loss, model.optimizer],
                     feed_dict = { model.ans_input: ans_in,
                                   model.ans_lens: ans_lens_in,
                                   model.ques_input: ques_in,
                                   model.ques_lens: ques_lens_in,
                                   model.labels: labels, model.keep_prob:
                                   config['drop_emb'] })

          train_loss_sum += train_loss
          print "[Average loss : %.5f]" % (train_loss_sum/(i+1)),

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
          print "\r[Dev] - [%.2f s left] - [%d of %d batches] " %\
            ((time.time()-dev_start_t)*(len(dev_order)-i-1)/(i+1), i+1, len(dev_order)),
          sys.stdout.flush()

          # Prepare dev bath by computing lengths and padding
          dev_batch = dev[num:num+test_batch_size]
          ans_lens_in = [ len(example[1]) for example in dev_batch ]
          ques_lens_in = [ len(example[0]) for example in dev_batch ]
          max_ans_len = max(ans_lens_in)
          max_ques_len = max(ques_lens_in)

          ans_in = [ pad(example[1], 0, max_ans_len) for example in dev_batch ]
          ques_in = [ pad(example[0], 0, max_ques_len) for example in dev_batch ]
          labels = [ example[2] for example in dev_batch ]

          dev_loss, predictions =\
            sess.run([model.loss, model.predictions],
                     feed_dict = { model.ans_input: ans_in,
                                   model.ans_lens: ans_lens_in,
                                   model.ques_input: ques_in,
                                   model.ques_lens: ques_lens_in,
                                   model.labels: labels,
                                   model.keep_prob: 1.0 })

          dev_loss_sum += dev_loss
          print "[Average loss : %.5f]" % (dev_loss_sum/(i+1)),

          # Compute overall prediction-error, error for 0s, and error for 1s
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

