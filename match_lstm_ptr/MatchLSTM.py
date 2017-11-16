import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.autograd import Variable

class MatchLSTM(nn.Module):
  ''' Match-LSTM model definition. Properties specified in config.'''

  # Constructor
  def __init__(self, config):
    # Call constructor of nn module.
    super(MatchLSTM, self).__init__()

    # Set-up parameters from config.
    self.load_from_config(config)

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

    self.dropoutp = nn.Dropout(self.dropout)
    self.dropoutq = nn.Dropout(self.dropout)
    
    # Passage and Question LSTMs (matrices Hp and Hq respectively).
    self.preprocessing_lstm = nn.LSTMCell(input_size = self.embed_size,
                                          hidden_size = self.hidden_size)

    # Attention transformations (variable names below given against those in
    # Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm
    # and answer pointer." arXiv preprint arXiv:1608.07905 (2016).)
    self.attend_question = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
    self.attend_passage = nn.Linear(self.hidden_size, self.hidden_size)
    self.attend_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
    self.alpha_transform = nn.Linear(self.hidden_size, 1)

    # Final Match-LSTM cells (bi-directional).
    self.match_lstm = nn.LSTMCell(input_size = 2 * self.hidden_size,
                                  hidden_size = self.hidden_size)

    # Answer pointer attention transformations.
    self.attend_match_lstm = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False)
    self.attend_answer = nn.Linear(self.hidden_size, self.hidden_size)
    self.beta_transform = nn.Linear(self.hidden_size, 1)
    self.dropout_ptr = nn.Dropout(self.dropout/2)

    # Answer pointer LSTM.
    self.answer_pointer_lstm = nn.LSTMCell(input_size = 2 * self.hidden_size,
                                           hidden_size = self.hidden_size)

  # Load configuration options
  def load_from_config(self, config):
    self.embed_size = config['embed_size']
    self.vocab_size = config['vocab_size']
    self.hidden_size = config['hidden_size']
    self.lr_rate = config['lr']
    self.glove_path = config['glove_path']
    self.optimizer = config['optimizer']
    self.index_to_word = config['index_to_word']
    self.word_to_index = config['word_to_index']
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


  # inp.shape = (seq_len, batch)
  # output.shape = (seq_len, batch, embed_size)
  def get_glove_embeddings(self, inp):
    output = np.zeros((inp.shape[0], inp.shape[1], self.embed_size))
    for i, batch in enumerate(inp):
      for j, word_id in enumerate(batch):
        output[i][j] = self.embedding[word_id]
    return self.placeholder(output)

  # Forward pass method.
  # passage = tuple((seq_len, batch), len_within_batch)
  # question = tuple((seq_len, batch), len_within_batch)
  # answer = tuple((2, batch))
  def forward(self, passage, question, answer):
    if not self.use_glove:
      padded_passage = self.placeholder(passage[0], False)
      padded_question = self.placeholder(question[0], False)
    batch_size = passage[0].shape[1]
    max_passage_len = passage[0].shape[0]
    max_question_len = question[0].shape[0]
    passage_lens = passage[1]
    question_lens = question[1]

    # Get embedded passage and question representations.
    if not self.use_glove:
      p = torch.transpose(self.embedding(torch.t(padded_passage)), 0, 1)
      q = torch.transpose(self.embedding(torch.t(padded_question)), 0, 1)
    else:
      p = self.get_glove_embeddings(passage[0])
      q = self.get_glove_embeddings(question[0])

    p = self.dropoutp(p)
    q = self.dropoutq(q)

    # Preprocessing LSTM outputs.
    Hp = []
    Hq = []

    # Pre-processing passage.
    hp, cp = self.get_initial_lstm(batch_size)
    for t in range(max_passage_len):
      hp, cp = self.preprocessing_lstm(p[t], (hp, cp))
      # Don't use LSTM output gating.
      hp = f.tanh(cp)
      mask = self.placeholder(np.array([ [1.0] if t < passage_lens[i] else [0.0] \
                                           for i in range(batch_size) ]))
      hp = hp * mask
      cp = cp * mask
      Hp.append(hp)

    # Pre-processing question.
    hq, cq = self.get_initial_lstm(batch_size)
    for t in range(max_question_len):
      hq, cq = self.preprocessing_lstm(q[t], (hq, cq))
      # Don't use LSTM output gating.
      hq = f.tanh(cq)
      mask = self.placeholder(np.array([ [1.0] if t < question_lens[i] else [0.0] \
                                           for i in range(batch_size) ]))
      hq = hq * mask
      cq = cq * mask
      Hq.append(hq)

    # H{p,q}.shape = (seq_len, batch, hdim)
    Hp = torch.stack(Hp, dim=0)
    Hq = torch.stack(Hq, dim=0)

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

    Hr = self.dropout_ptr(Hr)
    # attended_match_lstm.shape = (seq_len, batch, hdim)
    attended_match_lstm = self.attend_match_lstm(Hr)
    # {h,c}a.shape = (1, batch, hdim)
    ha, ca = self.get_initial_lstm(batch_size)
    answer_distributions = []
    losses = []
    for k in range(2):
      # Fk.shape = (seq_len, batch, hdim)
      Fk = f.tanh(attended_match_lstm + self.attend_answer(ha))

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

