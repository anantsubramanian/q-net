from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import sys
import tensorflow as tf

class QA_Model:
  def __init__(self, config):
    # Load configuration options
    embed_size = config['embed_size']
    nwords = config['nwords']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    lr_rate = config['lr']
    unequal_neg = config['unequal_neg']
    optimizer = config['optimizer']
    word2vec_path = config['word2vec_path']
    index_to_word = config['index_to_word']
    output_dim = 1
    incorrect_ratio = float(config['incorrect_ratio'])

    self.passage_input = tf.placeholder(tf.int32, [None, None], name="passage_input")
    self.passage_lens = tf.placeholder(tf.int32, [None], name='passage_lens')
    self.ans_input = tf.placeholder(tf.int32, [None, None], name="ans_input")
    self.ans_lens = tf.placeholder(tf.int32, [None], name='ans_lens')
    self.ques_input = tf.placeholder(tf.int32, [None, None], name="ques_input")
    self.ques_lens = tf.placeholder(tf.int32, [None], name='ques_lens')

    # Use commented with sparse cross entropy loss.
    #self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.labels = tf.placeholder(tf.float32, [None], name='labels')
    self.keep_prob = tf.placeholder(tf.float32)

    # Look-up question and answer embeddings. Use word2vec initialization if provided.
    with tf.variable_scope("embedding"):
      if word2vec_path:
        print "Loading word2vec vectors."
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        print "Done."
        sys.stdout.flush()
        assert index_to_word is not None

        # Create initial embedding matrix
        embedding = []
        not_found = 0
        for word in index_to_word:
          if word in word2vec.wv:
            embedding.append(word2vec.wv[word])
          else:
            not_found += 1
            print word
            embedding.append(np.array([0.0] * embed_size))

        print "Did not find %d/%d words not found in word2vec" % (not_found, nwords)
        embedding = np.array(embedding)
        self.embedding = tf.get_variable("embedding", shape=embedding.shape,
                                         initializer=tf.constant_initializer(embedding),
                                         trainable=True)
      else:
        self.embedding = tf.get_variable("embedding", [nwords, embed_size])

      self.ans_embs = tf.nn.embedding_lookup(self.embedding, self.ans_input)
      self.ques_embs = tf.nn.embedding_lookup(self.embedding, self.ques_input)
      self.passage_embs = tf.nn.embedding_lookup(self.embedding, self.passage_input)

    # Get answer encoded representation
    with tf.variable_scope("passage_rnn"):
      self.passage_cell = tf.contrib.rnn.LSTMCell(hidden_size)
      # Uncomment to use dropout
      #self.passage_cell = tf.contrib.rnn.DropoutWrapper(self.passage_cell,
      #                                              output_keep_prob=self.keep_prob)
      self.passage_cell = tf.contrib.rnn.MultiRNNCell([self.passage_cell] * num_layers)
      self.passage_rnn_outputs, _ = tf.nn.dynamic_rnn(self.passage_cell, self.passage_embs,
                                                  sequence_length=self.passage_lens,
                                                  dtype=tf.float32)
      passage_outputs = tf.squeeze(self.passage_rnn_outputs[:, -1, :])
      # Uncomment below to non-linearly project encoded answer representation
      #ans_outputs = tf.contrib.layers.fully_connected(
      #  activation_fn=tf.nn.relu, inputs=ans_outputs, num_outputs=embed_size,
      #  scope="fully_connect_ans",
      #  biases_initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope("answer_rnn"):
      self.ans_cell = tf.contrib.rnn.LSTMCell(hidden_size)
      # Uncomment to use dropout
      #self.ans_cell = tf.contrib.rnn.DropoutWrapper(self.ans_cell,
      #                                              output_keep_prob=self.keep_prob)
      self.ans_cell = tf.contrib.rnn.MultiRNNCell([self.ans_cell] * num_layers)
      self.ans_rnn_outputs, _ = tf.nn.dynamic_rnn(self.ans_cell, self.ans_embs,
                                                  sequence_length=self.ans_lens,
                                                  dtype=tf.float32)
      ans_outputs = tf.squeeze(self.ans_rnn_outputs[:, -1, :])
      # Uncomment below to non-linearly project encoded answer representation
      #ans_outputs = tf.contrib.layers.fully_connected(
      #  activation_fn=tf.nn.relu, inputs=ans_outputs, num_outputs=embed_size,
      #  scope="fully_connect_ans",
      #  biases_initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope("question_rnn"):
      self.ques_cell = tf.contrib.rnn.LSTMCell(hidden_size)
      #self.ques_cell = tf.contrib.rnn.DropoutWrapper(self.ques_cell,
      #                                               output_keep_prob=self.keep_prob)
      self.ques_cell = tf.contrib.rnn.MultiRNNCell([self.ques_cell] * num_layers)
      self.ques_rnn_outputs, _ = tf.nn.dynamic_rnn(self.ques_cell, self.ques_embs,
                                                   sequence_length=self.ques_lens,
                                                   dtype=tf.float32)
      ques_outputs = tf.squeeze(self.ques_rnn_outputs[:, -1, :])
      # Uncomment below to non-linearly project encoded question representation
      #outputs = tf.contrib.layers.fully_connected(
      #  activation_fn=tf.nn.relu, inputs=outputs, num_outputs=embed_size,
      #  scope = "fully_connect_ques",
      #  biases_initializer = tf.contrib.layers.xavier_initializer())

    # Combine question and answer embeddings by concatenation
    self.comb_embs = tf.reshape(tf.concat([passage_outputs, ans_outputs, ques_outputs], 1), [-1, 2*hidden_size])

    # Uncomment below to non-linearly project combined ans+ques embeddings
    self.comb_embs = tf.contrib.layers.fully_connected(
      activation_fn=tf.nn.tanh, inputs=self.comb_embs, num_outputs=2*hidden_size,
      scope="fully_connect_combined")

    # Linearly project combined embeddings to output space
    with tf.variable_scope("affine"):
      self.W_sm = tf.get_variable("affine_W", shape=[2*hidden_size, output_dim])
      #self.W_sm = tf.Variable(tf.random_uniform([2*hidden_size, output_dim]))
      self.b_sm = tf.Variable(tf.zeros([output_dim]))
      self.logits = tf.matmul(self.comb_embs, self.W_sm) + self.b_sm

    # Get the 0-1 prediction value
    # Uncomment below  instead for output_dim of 2
    #self.predictions = tf.argmax(self.logits, axis=1)
    self.predictions = tf.round(tf.nn.sigmoid(tf.squeeze(self.logits)))

    # Compute loss. More weight is given to the positive label examples if the
    # 'unequal_neg' flag is set.
    with tf.variable_scope("loss"):
      # Uncomment below to use sparse softmax cross entropy with output_dim as 2
      #self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
      # Uncomment below for hinge loss
      #self.losses =\
      #  tf.losses.hinge_loss(tf.expand_dims(self.labels,-1),self.logits)
      self.losses =\
        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                labels=tf.expand_dims(self.labels,-1))
      if unequal_neg:
        # incorrect_ratio extra weight is given to positive samples.
        self.losses = \
          tf.losses.compute_weighted_loss(self.losses,
                                          1.0 + tf.expand_dims(self.labels, -1) *\
                                          (incorrect_ratio-1))
        #self.losses = tf.add(self.losses, tf.multiply(self.losses, incorrect_ratio *\
        #                                              tf.to_float(self.labels)))
      self.loss = tf.reduce_mean(self.losses)

    # Decide between Adam, Momentum and SGD optimizers
    if optimizer == 'Adam':
      self.optimizer = tf.train.AdamOptimizer(lr_rate).minimize(self.loss)
    elif optimizer == 'Momentum':
      self.optimizer = tf.train.MomentumOptimizer(lr_rate, 0.9).minimize(self.loss)
    elif optimizer == 'SGD':
      self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

