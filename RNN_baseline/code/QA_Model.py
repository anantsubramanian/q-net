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
    output_dim = 2
    incorrect_ratio = float(config['incorrect_ratio'])

    self.ans_input = tf.placeholder(tf.int32, [None, None], name="ans_input")
    self.ans_lens = tf.placeholder(tf.int32, [None], name='ans_lens')
    self.ques_input = tf.placeholder(tf.int32, [None, None], name="ques_input")
    self.ques_lens = tf.placeholder(tf.int32, [None], name='ques_lens')

    self.labels = tf.placeholder(tf.int32, [None], name='labels')
    self.keep_prob = tf.placeholder(tf.float32)

    # Look-up question and answer embeddings
    with tf.variable_scope("embedding"):
      self.embedding = tf.get_variable("embedding", [nwords, embed_size])
      self.ans_embs = tf.nn.embedding_lookup(self.embedding, self.ans_input)
      self.ques_embs = tf.nn.embedding_lookup(self.embedding, self.ques_input)

    # Get answer encoded representation
    with tf.variable_scope("answer_rnn"):
      self.ans_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0)
      self.ans_cell = tf.contrib.rnn.DropoutWrapper(self.ans_cell,
                                                    output_keep_prob=self.keep_prob)
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
      self.ques_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0)
      self.ques_cell = tf.contrib.rnn.DropoutWrapper(self.ques_cell,
                                                     output_keep_prob=self.keep_prob)
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
    self.comb_embs = tf.concat([ans_outputs, ques_outputs], 1)

    # Uncomment below to non-linearly project combined ans+ques embeddings
    #self.comb_embs = tf.contrib.layers.fully_connected(
    #  activation_fn = tf.nn.relu, inputs=self.comb_embs, num_outputs=hidden_size,
    #  scope="fully_connect_combined",
    #  biases_initializer=tf.contrib.layers.xavier_initializer())

    # Linearly project combined embeddings to output space
    with tf.variable_scope("affine"):
      self.W_sm = tf.Variable(tf.random_uniform([2*hidden_size, output_dim]))
      self.b_sm = tf.Variable(tf.random_uniform([output_dim]))
      self.logits = tf.matmul(self.comb_embs, self.W_sm) + self.b_sm

    # Compute loss. More weight is given to the positive label examples if the
    # 'unequal_neg' flag is set.
    with tf.variable_scope("loss"):
      self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                   labels=self.labels)
      if unequal_neg:
        # incorrect_ratio extra weight is given to positive samples.
        self.losses = tf.add(self.losses, tf.multiply(self.losses, incorrect_ratio *\
                                                      tf.to_float(self.labels)))
      self.loss = tf.reduce_mean(self.losses)

    # Decide between Adam, Momentum and SGD optimizers
    if optimizer == 'Adam':
      self.optimizer = tf.train.AdamOptimizer(lr_rate).minimize(self.loss)
    elif optimizer == 'Momentum':
      self.optimizer = tf.train.MomentumOptimizer(lr_rate, 0.9).minimize(self.loss)
    elif optimizer == 'SGD':
      self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

