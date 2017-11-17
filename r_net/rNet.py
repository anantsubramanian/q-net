import numpy as np
import ipdb as pdb
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.autograd import Variable

class rNet(nn.Module):
  ''' Match-LSTM model definition. Properties specified in config.'''

  # Constructor
  def __init__(self, config):
    # Call constructor of nn module.
    super(rNet, self).__init__()

    # Set-up parameters from config.
    self.load_from_config(config)

    # Char Embedding look-up
    self.char_embedding = nn.Embedding(self.char_vocab_size, self.embed_size//2,
                                       self.char_to_index['<pad>'])
    # Embedding look-up.
    self.oov_count = 0
    self.oov_list = []
    if self.use_glove:
      embeddings = np.zeros((self.vocab_size, self.embed_size))
      with open(self.glove_path) as f:
        for line in f:
          line = line.split()
          if line[0] in self.word_to_index:
            embeddings[self.word_to_index[line[0]]] = np.array(map(float,line[1:]))
      for i, embedding in enumerate(embeddings):
        if sum(embedding) == 0:
          self.oov_count += 1
          self.oov_list.append(self.index_to_word[i])
      self.embedding = embeddings
    else:
      self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                    self.word_to_index['<pad>'])

    # Passage and Question character level GRUs.
    self.char_gru = nn.GRU(input_size = self.embed_size//2,
                           hidden_size = self.hidden_size,
                           num_layers = 1)

    #self.word_charlstm_f = nn.GRU(input_size = self.embed_size,
    #                            hidden_size = self.hidden_size,
    #                            num_layers = 1)
    #self.word_charlstm_b = nn.GRU(input_size = self.embed_size,
    #                            hidden_size = self.hidden_size,
    #                            num_layers = 1)

    # Passage and Question GRUs (matrices Hp and Hq respectively).
    self.p_dropout = nn.Dropout(self.dropout)
    self.q_dropout = nn.Dropout(self.dropout)

    self.preprocess_gru = nn.GRU(input_size = self.embed_size + 2*self.hidden_size,
                                 hidden_size = self.hidden_size,
                                 num_layers = 3, dropout=self.dropout)

    #self.passage_lstm_f = nn.GRU(input_size = self.embed_size,
    #                            hidden_size = self.hidden_size,
    #                            num_layers = 3)
    #self.question_lstm_b = nn.GRU(input_size = self.embed_size,
    #                             hidden_size = self.hidden_size,
    #                             num_layers = 3)
    #self.passage_lstm_b = nn.GRU(input_size = self.embed_size,
    #                            hidden_size = self.hidden_size,
    #                            num_layers = 3)
    #self.question_lstm_b = nn.GRU(input_size = self.embed_size,
    #                             hidden_size = self.hidden_size,
    #                             num_layers = 3)

    # Attention transformations (variable names below given against those in
    # Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm
    # and answer pointer." arXiv preprint arXiv:1608.07905 (2016).)
    self.attend_question = nn.Linear(2*self.hidden_size, self.hidden_size, bias = False)
    self.attend_passage = nn.Linear(2*self.hidden_size, self.hidden_size, bias = False)
    self.attend_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
    self.alpha_transform = nn.Linear(self.hidden_size, 1, bias = False)

    # Attention gating for MatchLSTM.
    self.gate_match_attention = nn.Linear(4 * self.hidden_size,
                                          4 * self.hidden_size,
                                          bias = False)

    # Final Match-LSTM cells (bi-directional).
    self.match_lstm = nn.LSTMCell(input_size = 4 * self.hidden_size,
                                  hidden_size = self.hidden_size)
    #self.match_lstm_forward = nn.LSTMCell(input_size = 2 * self.hidden_size,
    #                                      hidden_size = self.hidden_size)
    #self.match_lstm_backward = nn.LSTMCell(input_size = 2 * self.hidden_size,
    #                                       hidden_size = self.hidden_size)

    # Self-matching module parameters.
    self.match_lstm_dropout = nn.Dropout(self.dropout)
    self.attend_self_passage = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
    self.attend_self_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
    self.gamma_transform = nn.Linear(self.hidden_size, 1, bias = False)

    # Attention gating for self-matching LSTM.
    self.gate_self_attention = nn.Linear(4 * self.hidden_size,
                                         4 * self.hidden_size,
                                         bias = False)

    # Final self-matching LSTM cells (bi-directional).
    self.self_lstm = nn.LSTMCell(input_size = 4 * self.hidden_size,
                                 hidden_size = self.hidden_size)

    # Answer pointer attention transformations.
    self.attend_self_lstm = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
    self.attend_answer = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
    self.beta_transform = nn.Linear(self.hidden_size, 1, bias = False)

    # Answer pointer LSTM.
    self.answer_dropout = nn.Dropout(self.dropout)
    self.answer_pointer_lstm = nn.LSTMCell(input_size = 2 * self.hidden_size,
                                           hidden_size = self.hidden_size)

  # Load configuration options
  def load_from_config(self, config):
    self.embed_size = config['embed_size']
    self.char_vocab_size = config['char_vocab_size']
    self.vocab_size = config['vocab_size']
    self.hidden_size = config['hidden_size']
    self.lr_rate = config['lr']
    self.glove_path = config['glove_path']
    self.optimizer = config['optimizer']
    self.index_to_word = config['index_to_word']
    self.index_to_char = config['index_to_char']
    self.word_to_index = config['word_to_index']
    self.char_to_index = config['char_to_index']
    self.use_glove = config['use_glove']
    self.use_cuda = config['cuda']
    self.dropout = config['dropout']

  def save(self, path, epoch):
    torch.save(self, path + "/epoch_" + str(epoch) + ".pt")

  def load(self, path, epoch):
    self = torch.load(path + "/epoch_" + str(epoch) + ".pt")
    self.passage_lstm.flatten_parameters()
    self.question_lstm.flatten_parameters()
    return self

  def free_memory(self):
    del self.loss

  # Calls torch nn utils rnn pack_padded_sequence.
  # For Question and Passage LSTMs.
  # Assume that the batch is sorted in descending order.
  def make_packed_data(self, inp, lengths):
    return torch.nn.utils.rnn.pack_padded_sequence(inp, lengths)

  # Calls torch nn utils rnn pad_packed_sequence.
  # Returns (padded_seq, lens)
  def make_padded_sequence(self, inp):
    return torch.nn.utils.rnn.pad_packed_sequence(inp)

  def variable(self, v):
    if self.use_cuda:
      return Variable(v, requires_grad = False).cuda()
    return Variable(v, requires_grad = False)

  def placeholder(self, np_var, to_float=True):
    if to_float:
      np_var = np_var.astype(np.float32)
    v = self.variable(torch.from_numpy(np_var))
    return v

  # Get an initial tuple of (h0, c0).
  # h0, c0 have dims (num_directions * num_layers, batch_size, hidden_size)
  def get_initial_lstm(self, batch_size, for_cell = True):
    if not for_cell:
      return (self.variable(torch.zeros(1, batch_size, self.hidden_size)),
              self.variable(torch.zeros(1, batch_size, self.hidden_size)))
    return (self.variable(torch.zeros(batch_size, self.hidden_size)),
            self.variable(torch.zeros(batch_size, self.hidden_size)))

  # Get an initial tuple of h0.
  # h0 has dims (num_directions * num_layers, batch_size, hidden_size)
  def get_initial_gru(self, batch_size, num_layers = 1, for_cell = False):
    if not for_cell:
      return self.variable(torch.zeros(num_layers, batch_size, self.hidden_size))
    return self.variable(torch.zeros(batch_size, self.hidden_size))

  # inp.shape = (seq_len, batch)
  # output.shape = (seq_len, batch, embed_size)
  def get_glove_embeddings(self, inp):
    output = np.zeros((inp.shape[0], inp.shape[1], self.embed_size))
    for i, batch in enumerate(inp):
      for j, word_id in enumerate(batch):
        output[i][j] = self.embedding[word_id]
    return self.placeholder(output)

  # Forward pass method.
  # char_word_p_f = tuple((seq_len , batch, max_word_len), len_per_word)
  # char_word_p_b = tuple((seq_len , batch, max_word_len), len_per_word)
  # char_word_q_f = tuple((seq_len , batch, max_word_len), len_per_word)
  # char_word_q_b = tuple((seq_len , batch, max_word_len), len_per_word)
  # passage_f = tuple((seq_len, batch), len_within_batch)
  # passage_b = tuple((seq_len, batch), len_within_batch)
  # question_f = tuple((seq_len, batch), len_within_batch)
  # question_b = tuple((seq_len, batch), len_within_batch)
  # answer = tuple((2, batch))
  def forward(self, char_word_p_f, char_word_p_b, char_word_q_f, char_word_q_b,
              passage_f, passage_b, question_f, question_b, answer):

    if not self.use_glove:
      padded_passage_f = self.placeholder(passage_f[0], False)
      padded_question_f = self.placeholder(question_f[0], False)
      padded_passage_b = self.placeholder(passage_b[0], False)
      padded_question_b = self.placeholder(question_b[0], False)

    batch_size = passage_f[0].shape[1]
    max_passage_len = passage_f[0].shape[0]
    max_question_len = question_b[0].shape[0]
    max_char_word_len_q = char_word_q_f[0].shape[2]
    max_char_word_len_p = char_word_p_f[0].shape[2]

    passage_lens = passage_f[1]
    question_lens = question_b[1]

    char_start = time.time()

    padded_char_word_q_f = self.placeholder(char_word_q_f[0], False).contiguous()
    padded_char_word_q_b = self.placeholder(char_word_q_b[0], False).contiguous()
    padded_char_word_p_f = self.placeholder(char_word_p_f[0], False).contiguous()
    padded_char_word_p_b = self.placeholder(char_word_p_b[0], False).contiguous()

    char_word_q_lens = char_word_q_f[1]
    char_word_p_lens = char_word_p_b[1]

    p_word_ids_f = passage_f[0].reshape(-1)
    p_uniq_w_f, p_uniq_widx_f, p_word_order_f = np.unique(p_word_ids_f,return_index=True, return_inverse=True)

    p_word_ids_b = passage_b[0].reshape(-1)
    p_uniq_w_b, p_uniq_widx_b, p_word_order_b = np.unique(p_word_ids_b,return_index=True, return_inverse=True)

    q_word_ids_f = question_f[0].reshape(-1)
    q_uniq_w_f, q_uniq_widx_f, q_word_order_f = np.unique(q_word_ids_f,return_index=True, return_inverse=True)

    q_word_ids_b = question_b[0].reshape(-1)
    q_uniq_w_b, q_uniq_widx_b, q_word_order_b = np.unique(q_word_ids_b,return_index=True, return_inverse=True)

    # Get character level embedding for the words in the question.
    # padded_char_word_q_{f,b}.shape = (total_words, max_word_len)
    padded_char_word_q_f = padded_char_word_q_f.view(-1, max_char_word_len_q)
    padded_char_word_q_b = padded_char_word_q_b.view(-1, max_char_word_len_q)
    uniq_padded_char_word_q_f = padded_char_word_q_f[q_uniq_widx_f]
    uniq_padded_char_word_q_b = padded_char_word_q_b[q_uniq_widx_b]
    uniq_char_word_q_lens = (char_word_q_lens.reshape((-1))-1)[q_uniq_widx_f]

    # char_word_emb_q_{f,b}.shape = (max_word_len, total_words, emb_size)
    total_words_q = uniq_padded_char_word_q_f.size()[0]
    char_word_emb_q_f = torch.transpose(self.char_embedding(uniq_padded_char_word_q_f), 0 , 1)
    char_word_emb_q_b = torch.transpose(self.char_embedding(uniq_padded_char_word_q_b), 0 , 1)

    # Hc_w_q_{f,b}.shape = (max_word_len, total_words, hidden_size)
    Hc_w_q_f, _ = self.char_gru(char_word_emb_q_f,
                                self.get_initial_gru(total_words_q, 1))
    Hc_w_q_b, _ = self.char_gru(char_word_emb_q_b,
                                self.get_initial_gru(total_words_q, 1))

    # Get only the last layer
    # Hc_w_p_{f,b}s.shape = (total_words, hidden_size)
    last_word_idxs = uniq_char_word_q_lens.reshape((-1))-1
    Hc_w_q_fs, Hc_w_q_bs = [], []
    for i, word_idx in enumerate(last_word_idxs):
      Hc_w_q_fs.append(Hc_w_q_f[word_idx,i,:])
      Hc_w_q_bs.append(Hc_w_q_b[word_idx,i,:])

    Hc_w_q_f = torch.stack(Hc_w_q_fs, dim=0)
    Hc_w_q_b = torch.stack(Hc_w_q_bs, dim=0)

    Hc_w_q_f = Hc_w_q_f[q_word_order_f]
    Hc_w_q_b = Hc_w_q_b[q_word_order_b]

    # Hc_w_q_{f,b}.shape = (seq_len, batch_size, hidden_size)
    Hc_w_q_f = Hc_w_q_f.view(-1, batch_size, self.hidden_size)
    Hc_w_q_b = Hc_w_q_b.view(-1, batch_size, self.hidden_size)

    # q_c_f.shape = (seq_len, batch_size, 2 * hidden_size)
    q_c_f = torch.cat((Hc_w_q_f, Hc_w_q_b), dim=-1)

    # Get character level embedding for the words in the passage.
    # padded_char_word_p_{f,b}.shape = (total_words, max_word_len)
    padded_char_word_p_f = padded_char_word_p_f.view(-1, max_char_word_len_p)
    padded_char_word_p_b = padded_char_word_p_b.view(-1, max_char_word_len_p)

    uniq_padded_char_word_p_f = padded_char_word_p_f[p_uniq_widx_f]
    uniq_padded_char_word_p_b = padded_char_word_p_b[p_uniq_widx_b]
    uniq_char_word_p_lens = (char_word_p_lens.reshape((-1))-1)[p_uniq_widx_f]

    # char_word_emb_p_{f,b}.shape = (max_word_len, total_words, emb_size)
    total_words_p = uniq_padded_char_word_p_f.size()[0]
    char_word_emb_p_f = torch.transpose(self.char_embedding(uniq_padded_char_word_p_f), 0 , 1)
    char_word_emb_p_b = torch.transpose(self.char_embedding(uniq_padded_char_word_p_b), 0 , 1)

    # Hc_w_p_{f,b}.shape = (max_word_len, total_words, hidden_size)
    Hc_w_p_f, _ = self.char_gru(char_word_emb_p_f,
                                self.get_initial_gru(total_words_p, 1))
    Hc_w_p_b, _ = self.char_gru(char_word_emb_p_b,
                                self.get_initial_gru(total_words_p, 1))
    # Get only the last layer
    # Hc_w_p_{f,b}s.shape = (total_words, hidden_size)
    last_word_idxs = uniq_char_word_p_lens.reshape((-1))-1
    Hc_w_p_fs, Hc_w_p_bs = [], []
    for i, word_idx in enumerate(last_word_idxs):
      Hc_w_p_fs.append(Hc_w_p_f[word_idx,i,:])
      Hc_w_p_bs.append(Hc_w_p_b[word_idx,i,:])

    Hc_w_p_f = torch.stack(Hc_w_p_fs, dim=0)
    Hc_w_p_b = torch.stack(Hc_w_p_bs, dim=0)

    Hc_w_p_f = Hc_w_p_f[p_word_order_f]
    Hc_w_p_b = Hc_w_p_b[p_word_order_b]

    # Hc_w_p_{f,b}.shape = (sep_len, batch_size, hidden_size)
    Hc_w_p_f = Hc_w_p_f.view(-1, batch_size, self.hidden_size)
    Hc_w_p_b = Hc_w_p_b.view(-1, batch_size, self.hidden_size)

    # p_c_f.shape = (sep_len, batch_size, 2 * hidden_size)
    p_c_f = torch.cat((Hc_w_p_f, Hc_w_p_b), dim=-1)

    # Reverse individual sequences, and pad at end for character-level word embeddings
    # of passage and question.
    p_c_b = []
    for idx in range(batch_size):
      pad_len = max_passage_len - passage_lens[idx]
      indexes = self.variable(torch.arange(passage_lens[idx]-1, -1, -1).long())
      p_c_b_idx = p_c_f[:,idx,:].index_select(0, indexes)
      if passage_lens[idx] < max_passage_len:
        zeros = self.variable(torch.zeros(pad_len, 2 * self.hidden_size))
        p_c_b_idx = torch.cat((p_c_b_idx, zeros), dim=0)
      p_c_b.append(p_c_b_idx)

    # p_c_b.shape = (seq_len, batch_size, 2 * hidden_size)
    p_c_b = torch.stack(p_c_b, dim=1)

    q_c_b = []
    for idx in range(batch_size):
      pad_len = max_question_len - question_lens[idx]
      indexes = self.variable(torch.arange(question_lens[idx]-1, -1, -1).long())
      q_c_b_idx = q_c_f[:,idx,:].index_select(0, indexes)
      if question_lens[idx] < max_question_len:
        zeros = self.variable(torch.zeros(pad_len, 2 * self.hidden_size))
        q_c_b_idx = torch.cat((q_c_b_idx, zeros), dim=0)
      q_c_b.append(q_c_b_idx)

    # q_c_b.shape = (seq_len, batch_size, 2 * hidden_size)
    q_c_b = torch.stack(q_c_b, dim=1)

    # Get embedded passage and question representations.
    if not self.use_glove:
      p_f = torch.transpose(self.embedding(torch.t(padded_passage_f)), 0, 1)
      q_f = torch.transpose(self.embedding(torch.t(padded_question_f)), 0, 1)
      p_b = torch.transpose(self.embedding(torch.t(padded_passage_b)), 0, 1)
      q_b = torch.transpose(self.embedding(torch.t(padded_question_b)), 0, 1)
    else:
      p_f = self.get_glove_embeddings(passage_f[0])
      q_f = self.get_glove_embeddings(question_f[0])
      p_b = self.get_glove_embeddings(passage_b[0])
      q_b = self.get_glove_embeddings(question_b[0])

    # {p,q}_combined_{f,b}.shape = (seq_len, batch_size, 2 * hidden_size + embed_size)
    p_combined_f = torch.cat((p_f, p_c_f), dim=-1)
    q_combined_f = torch.cat((q_f, q_c_f), dim=-1)
    p_combined_b = torch.cat((p_b, p_c_b), dim=-1)
    q_combined_b = torch.cat((q_b, q_c_b), dim=-1)

    # Dropout on pre-processing inputs.
    p_combined_f = self.p_dropout(p_combined_f)
    p_combined_b = self.p_dropout(p_combined_b)
    q_combined_f = self.q_dropout(q_combined_f)
    q_combined_b = self.q_dropout(q_combined_b)

    # Preprocessing GRU outputs.
    # H{p,q}_{f,b}.shape = (seq_len, batch, hdim)
    Hp_f, _ = self.preprocess_gru(p_combined_f,
                                  self.get_initial_gru(batch_size, 3))
    Hq_f, _ = self.preprocess_gru(q_combined_f,
                                  self.get_initial_gru(batch_size, 3))
    Hp_b, _ = self.preprocess_gru(p_combined_b,
                                  self.get_initial_gru(batch_size, 3))
    Hq_b, _ = self.preprocess_gru(q_combined_b,
                                  self.get_initial_gru(batch_size, 3))

    # H{p,q}.shape = (seq_len, batch, 2 * hdim)
    Hp = torch.cat((Hp_f, Hp_b), dim=-1)
    Hq = torch.cat((Hq_f, Hq_b), dim=-1)

    # Mask out padding hidden states for passage LSTM output.
    mask_ps = []
    for t in range(max_passage_len):
      mask_ps.append(np.array([ [1.0] if t < passage_lens[i] else [0.0] \
                                  for i in range(batch_size) ]))
    Hp = Hp * self.placeholder(np.array(mask_ps))

    # Mask out padding hidden states for question LSTM output.
    mask_qs = []
    for t in range(max_question_len):
      mask_qs.append(np.array([ [1.0] if t < question_lens[i] else [0.0] \
                                  for i in range(batch_size) ]))
    Hq = Hq * self.placeholder(np.array(mask_qs))

    # Bi-directional match-LSTM layer.
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (1, batch, hdim)
    hf, cf = self.get_initial_lstm(batch_size)
    hb, cb = self.get_initial_lstm(batch_size)

    # Get vectors zi for each i in passage.
    # Attended question is the same at each time step. Just compute it once.
    # attended_question.shape = (seq_len, batch, hdim)
    attended_question = self.attend_question(Hq)
    Hf, Hb = [], []
    for i in range(max_passage_len):
        forward_idx = i
        backward_idx = max_passage_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        attn_start = time.time()
        gf = f.tanh(attended_question + \
                (self.attend_passage(Hp[forward_idx]).expand_as(attended_question) + \
                 self.attend_hidden(hf)))
        gb = f.tanh(attended_question + \
                (self.attend_passage(Hp[backward_idx]).expand_as(attended_question) + \
                 self.attend_hidden(hb)))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = f.softmax(self.alpha_transform(gf), dim=0)
        alpha_b = f.softmax(self.alpha_transform(gb), dim=0)

        # Hp[{forward,backward}_idx].shape = (batch, hdim)
        # Hq = (seq_len, batch, hdim)
        # weighted_Hq_f.shape = (batch, hdim)
        weighted_Hq_f = torch.squeeze(torch.bmm(alpha_f.permute(1, 2, 0),
                                      torch.transpose(Hq, 0, 1)), dim=1)
        weighted_Hq_b = torch.squeeze(torch.bmm(alpha_b.permute(1, 2, 0),
                                      torch.transpose(Hq, 0, 1)), dim=1)

        # z{f,b}.shape = (batch, 2 * hdim)
        zf = torch.cat((Hp[forward_idx], weighted_Hq_f), dim=-1)
        zb = torch.cat((Hp[backward_idx], weighted_Hq_b), dim=-1)

        # Mask vectors for {z,h,c}{f,b}.
        mask_f = np.array([ [1.0] if forward_idx < passage_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_b = np.array([ [1.0] if backward_idx < passage_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_f = self.placeholder(mask_f)
        mask_b = self.placeholder(mask_b)
        zf = zf * mask_f
        zb = zb * mask_b

        # Gating the input to the MatchLSTM.
        gating_f = f.sigmoid(self.gate_match_attention(zf))
        gating_b = f.sigmoid(self.gate_match_attention(zb))
        zf = zf * gating_f
        zb = zb * gating_b

        # Take forward and backward LSTM steps, with zf and zb as inputs.
        hf, cf = self.match_lstm(zf, (hf, cf))
        hb, cb = self.match_lstm(zb, (hb, cb))

        # Back to initial zero states for padded regions.
        hf = hf * mask_f
        cf = cf * mask_f
        hb = hb * mask_b
        cb = cb * mask_b

        # Append hidden states to create Hf and Hb matrices.
        # h{f,b}.shape = (batch, hdim)
        Hf.append(hf)
        Hb.append(hb)

    # H{f,b}.shape = (seq_len, batch, hdim)
    Hb = Hb[::-1]
    Hf = torch.stack(Hf, dim=0)
    Hb = torch.stack(Hb, dim=0)

    # Hr.shape = (seq_len, batch, 2 * hdim)
    Hr = torch.cat((Hf, Hb), dim=-1)

    # Dropout output of MatchLSTM.
    self.match_lstm_dropout(Hr)

    # Bi-directional self-matching LSTM layer.
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (1, batch, hdim)
    hf, cf = self.get_initial_lstm(batch_size)
    hb, cb = self.get_initial_lstm(batch_size)

    # Get vectors zi for each i in passage.
    # Attended passage is the same at each time step. Just compute it once.
    # attended_passage.shape = (seq_len, batch, hdim)
    attended_passage = self.attend_self_passage(Hr)
    Hf, Hb = [], []
    for i in range(max_passage_len):
        forward_idx = i
        backward_idx = max_passage_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        gf = f.tanh(attended_passage + \
                (self.attend_passage(Hr[forward_idx]).expand_as(attended_passage)))
        gb = f.tanh(attended_passage + \
                (self.attend_passage(Hr[backward_idx]).expand_as(attended_passage)))

        # gamma_{f,g}.shape = (seq_len, batch, 1)
        gamma_f = f.softmax(self.gamma_transform(gf), dim=0)
        gamma_b = f.softmax(self.gamma_transform(gb), dim=0)

        # Hr[{forward,backward}_idx].shape = (batch, 2 * hdim)
        # Hr = (seq_len, batch, 2 * hdim)
        # weighted_Hr_f.shape = (batch, 2 * hdim)
        weighted_Hr_f = torch.squeeze(torch.bmm(gamma_f.permute(1, 2, 0),
                                      torch.transpose(Hr, 0, 1)), dim=1)
        weighted_Hr_b = torch.squeeze(torch.bmm(gamma_b.permute(1, 2, 0),
                                      torch.transpose(Hr, 0, 1)), dim=1)

        # z{f,b}.shape = (batch, 4 * hdim)
        zf = torch.cat((Hr[forward_idx], weighted_Hr_f), dim=-1)
        zb = torch.cat((Hr[backward_idx], weighted_Hr_b), dim=-1)

        # Mask vectors for {z,h,c}{f,b}.
        mask_f = np.array([ [1.0] if forward_idx < passage_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_b = np.array([ [1.0] if backward_idx < passage_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_f = self.placeholder(mask_f)
        mask_b = self.placeholder(mask_b)
        zf = zf * mask_f
        zb = zb * mask_b

        # Gating the input to the self-match LSTM.
        gating_f = f.sigmoid(self.gate_self_attention(zf))
        gating_b = f.sigmoid(self.gate_self_attention(zb))
        zf = zf * gating_f
        zb = zb * gating_b

        # Take forward and backward LSTM steps, with zf and zb as inputs.
        hf, cf = self.match_lstm(zf, (hf, cf))
        hb, cb = self.match_lstm(zb, (hb, cb))

        # Back to initial zero states for padded regions.
        hf = hf * mask_f
        cf = cf * mask_f
        hb = hb * mask_b
        cb = cb * mask_b

        # Append hidden states to create Hf and Hb matrices.
        # h{f,b}.shape = (batch, hdim)
        Hf.append(hf)
        Hb.append(hb)

    # H{f,b}.shape = (seq_len, batch, hdim)
    Hb = Hb[::-1]
    Hf = torch.stack(Hf, dim=0)
    Hb = torch.stack(Hb, dim=0)

    # Hr.shape = (seq_len, batch, 2 * hdim)
    Hr = torch.cat((Hf, Hb), dim=-1)

    # Answer pointer input dropout.
    Hr = self.answer_dropout(Hr)

    # attended_self_lstm.shape = (seq_len, batch, hdim)
    attended_self_lstm = self.attend_self_lstm(Hr)

    # {h,c}a.shape = (1, batch, hdim)
    ha, ca = self.get_initial_lstm(batch_size)
    answer_distributions = []
    losses = []
    for k in range(2):
      # Fk.shape = (seq_len, batch, hdim)
      Fk = f.tanh(attended_self_lstm + self.attend_answer(ha))

      # beta_k_scores.shape = (seq_len, batch, 1)
      beta_ks = []
      beta_k_scores = self.beta_transform(Fk)
      for idx in range(batch_size):
        beta_k_idx = f.softmax(beta_k_scores[:passage_lens[idx],idx,:], dim=0)
        if beta_k_idx.size()[0] < max_passage_len:
          diff = max_passage_len - beta_k_idx.size()[0]
          zeros = self.variable(torch.zeros((diff, 1)))
          beta_k_idx = torch.cat((beta_k_idx, zeros), dim=0)
        # beta_k_idx.shape = (max_seq_len, 1)
        beta_ks.append(beta_k_idx)
        losses.append(-torch.log(torch.squeeze(beta_k_idx[answer[k][idx]])))

      # beta_k.shape = (seq_len, batch, 1)
      beta_k = torch.stack(beta_ks, dim=1)

      # Store distribution over passage words for answer start/end.
      answer_distributions.append(torch.t(torch.squeeze(beta_k, dim=-1)))

      # weighted_Hr.shape = (batch, 2*hdim)
      weighted_Hr = torch.squeeze(torch.bmm(beta_k.permute(1, 2, 0),
                                  torch.transpose(Hr, 0, 1)), dim=1)

      # LSTM step.
      ha, ca = self.answer_pointer_lstm(weighted_Hr, (ha, ca))

    # Compute the loss.
    self.loss = sum(losses)/batch_size
    return answer_distributions

