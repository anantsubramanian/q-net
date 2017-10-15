import argparse
import time
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
  parser.add_argument('--embed_size', type=int, default=512)
  parser.add_argument('--hidden_size', type=int, default=200)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=0.1)
  return parser

args = init_parser().parse_args()
assert not (args.train_json == None and args.train_pickle == None)
assert not (args.dev_json == None and args.dev_pickle == None)

#--------------------------- Read train and dev data --------------------------#
train_data = Data(num_incorrect_candidates=args.num_incorrect_samples)
if args.train_json:
  train_data.read_from_file(args.train_json, args.max_train_articles)
  print "Vocab size is:", train_data.dictionary.size()
  print "Sample words:", train_data.dictionary.index_to_word[:10]
  if args.dump_pickles:
    assert not args.train_pickle == None
    train_data.dump_pickle(args.train_pickle)
else:
  train_data = train_data.read_from_pickle(args.train_pickle)

dev_data = Data(train_data.dictionary)
if args.dev_json:
  dev_data.read_from_file(args.dev_json, args.max_dev_articles)
  if args.dump_pickles:
    assert not args.dev_pickle == None
    dev_data.dump_pickle(args.dev_pickle)
else:
  dev_data = dev_data.read_from_pickle(args.dev_pickle)
#------------------------------------------------------------------------------#

#--------------------------------- Preprocess data ----------------------------#

#------------------------------------------------------------------------------#

#---------------------------------- Create model ------------------------------#
config = { 'embed_size' : args.embed_size,
           'nwords' : train_data.dictionary.size(),
           'hidden_size' : args.hidden_size,
           'num_layers' : args.num_layers,
           'lr' : args.learning_rate }
qa_model = QA_Model(config)
#------------------------------------------------------------------------------#

#--------------------------------- Train System ----------------------------#
with tf.Session() as sess:
  tf.global_variables_initializer().run()

  if config['cont'] > 0:
       print ("Continue from ckpt", config['cont'])
       saver.restore(sess, model_dir + 'model' + str(config['cont'])+'.ckpt')

  start_time=time.time()
  print('Starting training at %r' % (start_time))

  for ITER in range(config['cont']+1,nepoch):
      random.shuffle(train_order)
      start_t = time.time()
      train_losses = []
      for num in train_order:

          # train on sent
          train_batch = train[num: num+ batch_size]
          ans_lens_in = [len(example[0]) for example in train_batch ]
          ques_lens_in = [len(example[1]) for example in train_batch ]
          labels = [example[2] for example in test_batch ]
          
          ans_in =[pad(example[0], S, max(ans_lens_in)) for example in train_batch ]
          ques_in =[pad(example[1], S, max(ques_lens_in)) for example in train_batch ]

          train_loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.x_input: x_in, model.x_lens: x_lens_in, model.keep_prob : config['drop_emb']})
          train_losses.append(train_loss)

      print('ITER %d, Train Loss: %.4f in time: %.4f' % ( ITER, sum(train_losses) / len(train), (time.time() - start_t)))

      test_losses = []

      print('Testing on Dev Set')
      for num in test_order:
          test_batch = test[num: num+ batch_size]
          ans_lens_in = [len(example[0]) for example in test_batch ]
          ques_lens_in = [len(example[1]) for example in test_batch ]
          labels = [example[2] for example in test_batch ]
          
          ans_in =[pad(example[0], S, max(ans_lens_in)) for example in test_batch ]
          ques_in =[pad(example[1], S, max(ques_lens_in)) for example in test_batch ]

          test_loss = sess.run(model.loss, feed_dict={model.ans_input: ans_in, model.ans_lens: ans_lens_in, model.ques_input: ques_in, model.ques_lens_in: model.ques_in, model.labels:labels, model.keep_prob : 1.0})

          test_losses.append(test_loss)

      print('ITER %d, Dev Loss: %.4f in time: %.4f' % (ITER ,test_losses, (time.time() - dev_start)))

      save_path = saver.save(sess, model_dir + 'model' + str(ITER) + '.ckpt')
      print('Model SAVED for ITER - ' , ITER)

#------------------------------------------------------------------------------#
