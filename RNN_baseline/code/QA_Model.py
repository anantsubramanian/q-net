import tensorflow as tf

class QA_Model:
    def __init__(self,config):
        embed_size =config['embed_size']

        nwords = config['nwords']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        lr_rate = config['lr']

        self.ans_input = tf.placeholder(tf.int32, [None, None], name="ans_input")
        self.ans_lens = tf.placeholder(tf.int32, [None], name = 'ans_lens')

        self.ques_input = tf.placeholder(tf.int32, [None, None], name="ques_input")
        self.ques_lens = tf.placeholder(tf.int32, [None], name = 'ques_lens')


        self.labels = tf.placeholder(tf.int32, [None], name = 'labels')
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope("Embedding"):
            self.embedding = tf.get_variable("embedding", [nwords, embed_size])
            self.ans_embs = tf.nn.embedding_lookup(self.embedding, self.ans_input)
            self.ques_embs = tf.nn.embedding_lookup(self.embedding, self.ques_input)

        with tf.variable_scope("Answer_RNN"):
            self.ans_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0)
            self.ans_cell = tf.contrib.rnn.DropoutWrapper( self.ans_cell, output_keep_prob=self.keep_prob)
            self.ans_cell = tf.contrib.rnn.MultiRNNCell([self.ans_cell] * num_layers)
            self.ans_rnn_outputs, _ = tf.nn.dynamic_rnn(self.ans_cell, self.ans_embs, sequence_length=self.ans_lens, dtype=tf.float32)
            ans_outputs = tf.squeeze(self.ans_rnn_outputs[:, -1, :])
            #ans_outputs = tf.reshape(self.ans_rnn_outputs[:,-1], [-1, hidden_size],name='answer_RNN_output')
            #outputs = tf.contrib.layers.fully_connected(
            #  activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500,
            #  scope = "Fully_connect_2", biases_initializer = tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Question_RNN"):
            self.ques_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0)
            self.ques_cell = tf.contrib.rnn.DropoutWrapper( self.ques_cell, output_keep_prob=self.keep_prob)
            self.ques_cell = tf.contrib.rnn.MultiRNNCell([self.ques_cell] * num_layers)
            self.ques_rnn_outputs, _ = tf.nn.dynamic_rnn(self.ques_cell, self.ques_embs, sequence_length=self.ques_lens, dtype=tf.float32)
            ques_outputs = tf.squeeze(self.ques_rnn_outputs[:, -1, :])
            #ques_outputs = tf.reshape(self.ques_rnn_outputs[:,-1], [-1, hidden_size],name='question_RNN_output')
            #outputs = tf.contrib.layers.fully_connected(
            #  activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500,
            #  scope = "Fully_connect_2", biases_initializer = tf.contrib.layers.xavier_initializer())

        #Combine 
        self.comb_embs=tf.concat([ans_outputs,ques_outputs],1) #2nd dimension if including Batch .. 
 
        '''outputs = tf.contrib.layers.fully_connected(
            activation_fn = tf.nn.relu, inputs = self.comb_embs, num_outputs = 100,
            scope = "Fully_connect_1", biases_initializer = tf.contrib.layers.xavier_initializer())'''


        with tf.variable_scope("Affine"):
            self.W_sm = tf.Variable(tf.random_uniform([2 * hidden_size, 2]))
            self.b_sm = tf.Variable(tf.random_uniform([2]))
            self.logits = tf.matmul(self.comb_embs, self.W_sm) + self.b_sm
            #self.softmax = tf.nn.softmax(self.logits, dim=-1, name='softmax_final')

        with tf.variable_scope("Loss"):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(self.losses)

        if config['optimizer'] == 'Adam':
            if(lr_rate==0):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            else:
                self.optimizer = tf.train.AdamOptimizer(epsilon=lr_rate).minimize(self.loss)
        elif config['optimizer'] == 'Momentum':
            if(lr_rate==0):
                self.optimizer = tf.train.MomentumOptimizer(0.1,0.9).minimize(self.loss)
            else:
                self.optimizer = tf.train.MomentumOptimizer(lr_rate,0.9).minimize(self.loss)
        elif config['optimizer'] == 'SGD':
            if(lr_rate==0):
                self.optimizer = tf.train.GradientDescentOptimizer().minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

