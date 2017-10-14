import tensorflow as tf

class QA_Model:
    def __init__(self,config):
        ans_embed_size =config['embed_size']
        ques_embed_size =config['embed_size']

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

        with tf.variable_scope("Answer Embedding"):
            self.ans_embedding = tf.get_variable("ans embedding", [nwords, ans_embed_size])
            self.ans_embs = tf.nn.embedding_lookup(self.ans_embedding, self.ans_input)

        with tf.variable_scope("Question Embedding"):
            self.ques_embedding = tf.get_variable("ques embedding", [nwords, ques_embed_size])
            self.ques_embs = tf.nn.embedding_lookup(self.ques_embedding, self.ques_input)

        #emb combine 
        self.comb_embs=tf.concat([self.ans_embs,self.ques_embs],2) #2nd dimension if including Batch .. 
 
        #outputs = tf.expand_dims(self.comb_embs, -1)

        #with tf.variable_scope('conv1_0') as scope:
        #    kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 8], dtype=tf.float32, stddev=1e-1))
        #    outputs = tf.nn.conv2d(outputs, kernel, [1, 1, 1, 1], padding='SAME',  use_cudnn_on_gpu=True, name="conv0") # (B,T,F,1) -> (B,T,F,32)
        #    biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
        #                             trainable=True, name='biases_conv0')
        #    outputs= tf.nn.bias_add(outputs, biases)
        #    outputs = tf.nn.relu(outputs, name="relu_conv0")

        #outputs = tf.reshape(outputs,[tf.shape(outputs)[0],tf.shape(outputs)[1],(int((int(embed_size))*8))])

        outputs = tf.contrib.layers.fully_connected(
            activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500,
            scope = "Fully_connect_1", biases_initializer = tf.contrib.layers.xavier_initializer())


        #self.cnn_output_drop = tf.nn.dropout(outputs,self.keep_prob)


        #Batch major
        with tf.variable_scope("RNN"):
            self.cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=False)
            self.cell = tf.contrib.rnn.DropoutWrapper( self.cell, output_keep_prob=self.keep_prob)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * num_layers, state_is_tuple=False)
            self.rnn_outputs, _ = tf.nn.dynamic_rnn(self.cell, outputs, sequence_length=tf.add(self.ans_lens,self.ques_lens), dtype=tf.float32)
            outputs = tf.reshape(self.rnn_outputs[:,-1], [-1, hidden_size],name='rnn_output')
            outputs = tf.contrib.layers.fully_connected(
              activation_fn = tf.nn.relu, inputs = outputs, num_outputs = 500,
              scope = "Fully_connect_2", biases_initializer = tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("Affine"):
            self.W_sm = tf.Variable(tf.random_uniform([hidden_size, 2]))
            self.b_sm = tf.Variable(tf.random_uniform([2]))
            self.logits = tf.matmul(tf.squeeze(self.output), self.W_sm) + self.b_sm
            self.softmax = tf.nn.softmax(self.logits, dim=-1, name='softmax_final')

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
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate,0.9).minimize(self.loss)
        elif config['optimizer'] == 'SGD':
            if(lr_rate==0):
                self.optimizer = tf.train.GradientDescentOptimizer().minimize(self.loss)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)
