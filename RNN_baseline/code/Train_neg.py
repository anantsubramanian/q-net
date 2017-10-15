import argparse
import pdb
import random
import sys
import time
import tensorflow as tf
import numpy
from Input import Dictionary, Data
from QA_Model import QA_Model


def pad(seq, element, length):
    assert len(seq) <= length
    r = seq + [element] * (length - len(seq))
    assert len(r) == length
    return r

def init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_json', dest='train_json')
  parser.add_argument('--dev_json', dest='dev_json')
  parser.add_argument('--train_pickle', dest='train_pickle')
  parser.add_argument('--dev_pickle', dest='dev_pickle')
  parser.add_argument('--dump_pickles', action='store_true')
  parser.add_argument('--num_incorrect_samples', type=int, dest='num_incorrect_samples',
                      default=5)
  parser.add_argument('--max_train_articles', type=int, dest='max_train_articles',
                      default=-1)
  parser.add_argument('--max_dev_articles', type=int, dest='max_dev_articles',
                      default=-1)
  parser.add_argument('--embed_size', type=int, default=256)
  parser.add_argument('--hidden_size', type=int, default=300)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=0.05)
  parser.add_argument('--ckpt', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--model_dir', default='./testing')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--dropout_keep_value', type=float, default=1.0)
  parser.add_argument('--optimizer', default='Adam')
  return parser

args = init_parser().parse_args()
assert not (args.train_json == None and args.train_pickle == None)
assert not (args.dev_json == None and args.dev_pickle == None)

#--------------------------- Read train and dev data --------------------------#
train_data = Data(num_incorrect_candidates=args.num_incorrect_samples)
if args.train_json:
  train_data.read_from_file(args.train_json, args.max_train_articles)
  if args.dump_pickles:
    assert not args.train_pickle == None
    train_data.dump_pickle(args.train_pickle)
else:
  train_data = train_data.read_from_pickle(args.train_pickle)

print "Vocab size is:", train_data.dictionary.size()
print "Sample words:", train_data.dictionary.index_to_word[:10]

'''dev_data = Data(train_data.dictionary)
if args.dev_json:
  dev_data.read_from_file(args.dev_json, args.max_dev_articles)
  if args.dump_pickles:
    assert not args.dev_pickle == None
    dev_data.dump_pickle(args.dev_pickle)
else:
  dev_data = dev_data.read_from_pickle(args.dev_pickle)'''
dev_data = train_data
#------------------------------------------------------------------------------#

#--------------------------------- Preprocess data ----------------------------#

#------------------------------------------------------------------------------#

#---------------------------------- Create model ------------------------------#
config = { 'embed_size' : args.embed_size,
           'nwords' : train_data.dictionary.size(),
           'hidden_size' : args.hidden_size,
           'num_layers' : args.num_layers,
           'lr' : args.learning_rate,
           'cont': args.ckpt,
           'drop_emb': args.dropout_keep_value,
           'optimizer': args.optimizer }
model = QA_Model(config)
#------------------------------------------------------------------------------#

#--------------------------------- Train System ----------------------------#
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  saver = tf.train.Saver(max_to_keep=args.epochs)
  if config['cont'] > 0:
       print ("Continue from ckpt", config['cont'])
       saver.restore(sess, args.model_dir + 'model' + str(config['cont'])+'.ckpt')

  start_time=time.time()
  print('Starting training at %r' % (start_time))

  
  train = train_data.data
  test = dev_data.data
  batch_size = args.batch_size
  train.sort(cmp=lambda x,y: len(x[0]) + len(x[1]) - (len(y[0]) + len(y[1])))
  labels_l = numpy.array([example[2] for example in train ])
  train_q = numpy.array([example[0] for example in train ])
  train_a = numpy.array([example[1] for example in train ])
  pdb.set_trace()
  neg_index = numpy.where(labels_l==0)[0]
  pos_index = numpy.where(labels_l==1)[0]
  pos_train_ques = train_q[pos_index]
  pos_train_ans = train_a[pos_index]
  neg_train_ans = train_a[neg_index]
  neg_train_ques = train_q[neg_index] 
  train_order = [ i for i in range(0, 2*len(pos_train_ques), batch_size) ]
  test_order = [ i for i in range(0, len(test), batch_size) ]

  for ITER in range(config['cont']+1,args.epochs):
      neg_epoch_indices = numpy.random.randint(0,len(neg_train_ans),len(pos_train_ans))
      train_ques = numpy.concatenate((pos_train_ques ,neg_train_ques[neg_epoch_indices]))
      train_ans = numpy.concatenate((pos_train_ans, neg_train_ans[neg_epoch_indices]))
      train_labels = numpy.array([1]*len(pos_train_ans) + [0]*len(pos_train_ans))
      ind = [i for i in range(len(train_labels))]
      random.shuffle(ind)
      train_ques = train_ques[ind]
      train_ans = train_ans[ind] 
      train_labels = train_labels[ind]
      
      random.shuffle(train_order)
      start_t = time.time()
      train_loss_sum = 0.0
      for i, num in enumerate(train_order):
          # train on sent
          print "\r%d of %d batches processed " % (i+1, len(train_order)),
          print "(time left = %.2f seconds) : " % ((time.time()-start_t) * (len(train_order)-i-1) / (i+1)),
          sys.stdout.flush()
          train_ques_batch = train_ques[num: num+batch_size]
          train_ans_batch = train_ans[num: num+batch_size]          
          train_labels_batch = train_labels[num: num+batch_size]

          ans_lens_in = [len(example) for example in train_ans_batch ]
          ques_lens_in = [len(example) for example in train_ques_batch ]
          labels = [example for example in train_labels_batch ]
          
          ans_in =[pad(example, 0, max(ans_lens_in)) for example in train_ans_batch ]
          ques_in =[pad(example, 0, max(ques_lens_in)) for example in train_ques_batch ]

          train_loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.ans_input: ans_in, model.ans_lens: ans_lens_in, model.ques_input: ques_in, model.ques_lens: ques_lens_in, model.labels:labels, model.keep_prob : config['drop_emb']})
          train_loss_sum += train_loss
          print "Average loss = %.5f" % (train_loss_sum/(i+1)),

      print ""
      print('ITER %d, Train Loss: %.4f in time: %.4f' % ( ITER, train_loss_sum/len(train_order), (time.time() - start_t)))

      '''test_losses = []
      dev_start = time.time()
      print('Testing on Dev Set')
      for num in test_order:
          test_batch = test[num: num+ batch_size]
          ans_lens_in = [len(example[1]) for example in test_batch ]
          ques_lens_in = [len(example[0]) for example in test_batch ]
          labels = [example[2] for example in test_batch ]
          
          ans_in =[pad(example[1], 0, max(ans_lens_in)) for example in test_batch ]
          ques_in =[pad(example[0], 0, max(ques_lens_in)) for example in test_batch ]

          test_loss = sess.run(model.loss, feed_dict={model.ans_input: ans_in, model.ans_lens: ans_lens_in, model.ques_input: ques_in, model.ques_lens: ques_lens_in, model.labels:labels, model.keep_prob : 1.0})

          test_losses.append(test_loss)

      print('ITER %d, Dev Loss: %.4f in time: %.4f' % (ITER , sum(test_losses) / len(test), (time.time() - dev_start)))'''

      save_path = saver.save(sess, args.model_dir + 'model' + str(ITER) + '.ckpt')
      print('Model SAVED for ITER - ' , ITER)

#------------------------------------------------------------------------------#
