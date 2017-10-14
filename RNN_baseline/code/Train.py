import argparse

from Input import Dictionary, Data
from QA_Model import QA_Model

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

