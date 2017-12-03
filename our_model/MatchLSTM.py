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

    # Construct the model, storing all necessary layers.
    self.build_model()

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
    self.num_pos_tags = config['num_pos_tags']
    self.f1_loss_ratio = config['f1_loss_ratio']
    self.num_preprocessing_layers = config['num_preprocessing_layers']
    self.num_question_matchlstm_layers = config['num_question_matchlstm_layers']
    self.num_passage_matchlstm_layers = config['num_passage_matchlstm_layers']

  def build_model(self):
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

    # Passage and Question pre-processing LSTMs (matrices Hp and Hq respectively).
    for layer_no in range(self.num_preprocessing_layers):
      input_size = self.embed_size + self.num_pos_tags \
                     if layer_no == 0 else self.hidden_size * 2
      setattr(self, 'dropoutp_' + str(layer_no), nn.Dropout(self.dropout))
      setattr(self, 'dropoutq_' + str(layer_no), nn.Dropout(self.dropout))
      setattr(self, 'preprocessing_lstm_' + str(layer_no),
              nn.LSTMCell(input_size = input_size, hidden_size = self.hidden_size))

    # Passage-aware question MatchLSTM layer.
    for layer_no in range(self.num_question_matchlstm_layers):
      setattr(self, 'attend_passage_for_question_' + str(layer_no),
              nn.Linear(self.hidden_size * 2, self.hidden_size, bias = False))
      setattr(self, 'attend_question_for_question_' + str(layer_no),
              nn.Linear(self.hidden_size * 2, self.hidden_size))
      setattr(self, 'attend_question_hidden_' + str(layer_no),
              nn.Linear(self.hidden_size, self.hidden_size, bias = False))
      setattr(self, 'question_alpha_transform_' + str(layer_no),
              nn.Linear(self.hidden_size, 1))
      # Final Match-LSTM cells (bi-directional).
      setattr(self, 'question_match_lstm_' + str(layer_no),
              nn.LSTMCell(input_size = self.hidden_size * 4,
                          hidden_size = self.hidden_size))
      setattr(self, 'dropout_question_matchlstm_' + str(layer_no),
              nn.Dropout(self.dropout))

    # Attention transformations (variable names below given against those in
    # Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm
    # and answer pointer." arXiv preprint arXiv:1608.07905 (2016).)
    for layer_no in range(self.num_passage_matchlstm_layers):
      setattr(self, 'attend_question_for_passage_' + str(layer_no),
              nn.Linear(self.hidden_size * 4 + self.num_pos_tags, self.hidden_size,
                        bias = False))
      setattr(self, 'attend_passage_for_passage_' + str(layer_no),
              nn.Linear(self.hidden_size * 2, self.hidden_size))
      setattr(self, 'attend_passage_hidden_' + str(layer_no),
              nn.Linear(self.hidden_size, self.hidden_size, bias = False))
      setattr(self, 'passage_alpha_transform_' + str(layer_no),
              nn.Linear(self.hidden_size, 1))
      # Final Match-LSTM cells (bi-directional).
      setattr(self, 'passage_match_lstm_' + str(layer_no),
              nn.LSTMCell(input_size = self.hidden_size * 6 + self.num_pos_tags,
                          hidden_size = self.hidden_size))
      setattr(self, 'dropout_passage_matchlstm_' + str(layer_no), nn.Dropout(self.dropout))

    # Question attention for answer pointer network.
    self.attend_question_for_answer = nn.Linear(self.hidden_size * 4 + self.num_pos_tags,
                                                self.hidden_size, bias = False)
    self.alpha_transform_for_answer = nn.Linear(self.hidden_size, 1)

    # Answer pointer attention transformations.
    self.attend_match_lstm = nn.Linear(self.hidden_size * 2 + self.num_pos_tags,
                                       self.hidden_size * 4 + self.num_pos_tags,
                                       bias = False)
    self.attend_match_lstm_b = nn.Linear(self.hidden_size * 2 + self.num_pos_tags,
                                         self.hidden_size * 4 + self.num_pos_tags,
                                         bias = False)
    self.attend_answer = nn.Linear(self.hidden_size * 4 + self.num_pos_tags,
                                   self.hidden_size * 4 + self.num_pos_tags)
    self.beta_transform = nn.Linear(self.hidden_size * 4 + self.num_pos_tags, 1)

    # Answer pointer LSTM.
    self.answer_pointer_lstm = \
      nn.LSTMCell(input_size = self.hidden_size * 6 + self.num_pos_tags * 2,
                  hidden_size = self.hidden_size * 4 + self.num_pos_tags)

  def save(self, path, epoch):
    torch.save(self, path + "/epoch_" + str(epoch) + ".pt")

  def load(self, path, epoch):
    self = torch.load(path + "/epoch_" + str(epoch) + ".pt")
    return self

  def load_from_file(self, path):
    self = torch.load(path)
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
  # If for a cell, they have dims (batch_size, hidden_size)
  def get_initial_lstm(self, batch_size, for_cell = True, hidden_size = None):
    if hidden_size is None:
      hidden_size = self.hidden_size
    if not for_cell:
      return (self.variable(torch.zeros(1, batch_size, hidden_size)),
              self.variable(torch.zeros(1, batch_size, hidden_size)))
    return (self.variable(torch.zeros(batch_size, hidden_size)),
            self.variable(torch.zeros(batch_size, hidden_size)))

  # inp.shape = (seq_len, batch)
  # output.shape = (seq_len, batch, embed_size)
  def get_glove_embeddings(self, inp):
    output = np.zeros((inp.shape[0], inp.shape[1], self.embed_size))
    for i, batch in enumerate(inp):
      for j, word_id in enumerate(batch):
        output[i][j] = self.embedding[word_id]
    return self.placeholder(output)

  # Get hidden states of a bi-directional pre-processing LSTM run over
  # the given input sequence.
  def preprocess_input(self, layer_no, inputs, max_len, input_lens,
                       batch_size):
    Hf, Hb = [], []
    hf, cf = self.get_initial_lstm(batch_size)
    hb, cb = self.get_initial_lstm(batch_size)
    for t in range(max_len):
      t_b = max_len - t - 1
      hf, cf = getattr(self, 'preprocessing_lstm_' + layer_no)(inputs[t], (hf, cf))
      hb, cb = getattr(self, 'preprocessing_lstm_' + layer_no)(inputs[t_b], (hb, cb))

      # Don't use LSTM output gating.
      hf = f.tanh(cf)
      hb = f.tanh(cb)

      # Mask out padded regions of input.
      mask_f = self.placeholder(np.array([ [1.0] if t < input_lens[i] else [0.0] \
                                             for i in range(batch_size) ]))
      mask_b = self.placeholder(np.array([ [1.0] if t_b < input_lens[i] else [0.0] \
                                             for i in range(batch_size) ]))
      hf = hf * mask_f
      cf = cf * mask_f
      hb = hb * mask_b
      cb = cb * mask_b
      Hf.append(hf)
      Hb.append(hb)

    # H{f,b}.shape = (seq_len, batch, hdim)
    # H.shape = (seq_len, batch, 2 * hdim)
    Hb = Hb[::-1]
    Hf = torch.stack(Hf, dim=0)
    Hb = torch.stack(Hb, dim=0)
    H = torch.cat((Hf, Hb), dim=-1)
    return H

  # Get a passage-aware question representation.
  def match_passage_question(self, layer_no, Hqi, Hp, max_question_len,
                             question_lens, batch_size):
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (batch, hdim)
    hf, cf = self.get_initial_lstm(batch_size)
    hb, cb = self.get_initial_lstm(batch_size)

    # Get vectors zi for each i in passage.
    # Attended passage is the same at each time step. Just compute it once.
    # attended_passage.shape = (seq_len, batch, hdim)
    attended_passage = getattr(self, 'attend_passage_for_question_' + layer_no)(Hp)
    Hf, Hb = [], []
    for i in range(max_question_len):
        forward_idx = i
        backward_idx = max_question_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        gf = f.tanh(attended_passage + \
                (getattr(self, 'attend_question_for_question_' + layer_no)(Hqi[forward_idx]) + \
                 getattr(self, 'attend_question_hidden_' + layer_no)(hf)))
        gb = f.tanh(attended_passage + \
                (getattr(self, 'attend_question_for_question_' + layer_no)(Hqi[backward_idx]) + \
                 getattr(self, 'attend_question_hidden_' + layer_no)(hb)))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = f.softmax(getattr(self, 'question_alpha_transform_' + layer_no)(gf),
                            dim=0)
        alpha_b = f.softmax(getattr(self, 'question_alpha_transform_' + layer_no)(gb),
                            dim=0)

        # Hq[{forward,backward}_idx].shape = (batch, 2 * hdim)
        # Hp = (seq_len, batch, 2 * hdim)
        # weighted_Hp_f.shape = (batch, 2 * hdim)
        weighted_Hp_f = torch.squeeze(torch.bmm(alpha_f.permute(1, 2, 0),
                                      torch.transpose(Hp, 0, 1)), dim=1)
        weighted_Hp_b = torch.squeeze(torch.bmm(alpha_b.permute(1, 2, 0),
                                      torch.transpose(Hp, 0, 1)), dim=1)

        # z{f,b}.shape = (batch, 4 * hdim)
        zf = torch.cat((Hqi[forward_idx], weighted_Hp_f), dim=-1)
        zb = torch.cat((Hqi[backward_idx], weighted_Hp_b), dim=-1)

        # Mask vectors for {z,h,c}{f,b}.
        mask_f = np.array([ [1.0] if forward_idx < question_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_b = np.array([ [1.0] if backward_idx < question_lens[i] else [0.0] \
                              for i in range(batch_size) ])
        mask_f = self.placeholder(mask_f)
        mask_b = self.placeholder(mask_b)
        zf = zf * mask_f
        zb = zb * mask_b

        # Take forward and backward LSTM steps, with zf and zb as inputs.
        hf, cf = getattr(self, 'question_match_lstm_' + layer_no)(zf, (hf, cf))
        hb, cb = getattr(self, 'question_match_lstm_' + layer_no)(zb, (hb, cb))

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
    return Hr

  # Get a question-aware passage representation.
  def match_question_passage(self, layer_no, Hpi, Hq, max_passage_len,
                             passage_lens, batch_size):
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (batch, hdim)
    hf, cf = self.get_initial_lstm(batch_size)
    hb, cb = self.get_initial_lstm(batch_size)

    # Get vectors zi for each i in passage.
    # Attended question is the same at each time step. Just compute it once.
    # attended_question.shape = (seq_len, batch, hdim)
    attended_question = getattr(self, 'attend_question_for_passage_' + layer_no)(Hq)
    Hf, Hb = [], []
    for i in range(max_passage_len):
        forward_idx = i
        backward_idx = max_passage_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        gf = f.tanh(attended_question + \
                (getattr(self, 'attend_passage_for_passage_' + layer_no)(Hpi[forward_idx]) + \
                 getattr(self, 'attend_passage_hidden_' + layer_no)(hf)))
        gb = f.tanh(attended_question + \
                (getattr(self, 'attend_passage_for_passage_' + layer_no)(Hpi[backward_idx]) + \
                 getattr(self, 'attend_passage_hidden_' + layer_no)(hb)))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = f.softmax(getattr(self, 'passage_alpha_transform_' + layer_no)(gf), dim=0)
        alpha_b = f.softmax(getattr(self, 'passage_alpha_transform_' + layer_no)(gb), dim=0)

        # Hp[{forward,backward}_idx].shape = (batch, 2 * hdim)
        # Hq = (seq_len, batch, 2 * hdim)
        # weighted_Hq_f.shape = (batch, 2 * hdim)
        weighted_Hq_f = torch.squeeze(torch.bmm(alpha_f.permute(1, 2, 0),
                                      torch.transpose(Hq, 0, 1)), dim=1)
        weighted_Hq_b = torch.squeeze(torch.bmm(alpha_b.permute(1, 2, 0),
                                      torch.transpose(Hq, 0, 1)), dim=1)

        # z{f,b}.shape = (batch, 4 * hdim)
        zf = torch.cat((Hpi[forward_idx], weighted_Hq_f), dim=-1)
        zb = torch.cat((Hpi[backward_idx], weighted_Hq_b), dim=-1)

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
        hf, cf = getattr(self, 'passage_match_lstm_' + layer_no)(zf, (hf, cf))
        hb, cb = getattr(self, 'passage_match_lstm_' + layer_no)(zb, (hb, cb))

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
    return Hr

  # Boundary pointer model, that gives probability distributions over the
  # answer start and answer end indices. Additionally returns the loss
  # for training.
  def answer_pointer(self, Hr, Hp, Hq, max_question_len, question_lens,
                     max_passage_len, passage_lens, batch_size, answer,
                     f1_matrices):
    # attended_match_lstm[_b].shape = (seq_len, batch, 4*hdim + num_pos_tags)
    attended_match_lstm = self.attend_match_lstm(Hr)
    attended_match_lstm_b = self.attend_match_lstm_b(Hr)

    # weighted_Hq_for_answer.shape = (batch, 4*hdim + num_pos_tags)
    attended_question_for_answer = f.tanh(self.attend_question_for_answer(Hq))
    alpha_q = f.softmax(self.alpha_transform_for_answer(attended_question_for_answer), dim=0)
    weighted_Hq_for_answer = torch.squeeze(torch.bmm(alpha_q.permute(1, 2, 0),
                                                     torch.transpose(Hq, 0, 1)), dim=1)

    # {h,c}{a,b}.shape = (batch, 4*hdim + num_pos_tags)
    ha, ca = self.get_initial_lstm(batch_size, hidden_size=self.hidden_size * 4 + self.num_pos_tags)
    hb, cb = self.get_initial_lstm(batch_size, hidden_size=self.hidden_size * 4 + self.num_pos_tags)

    answer_distributions = []
    answer_distributions_b = []
    mle_losses = []

    # Two three-step LSTMs: Point to the start of the answer first, then the end,
    # and point to the answer end, then the start.
    # 1st step initializes the hidden states to some answer representations.
    # 2nd predicts start/end distribution.
    # 3rd predicts end/start distribution.
    for k in range(3):
      # Fk[_b].shape = (seq_len, batch, 4*hdim + num_pos_tags)
      Fk = f.tanh(attended_match_lstm + self.attend_answer(ha) + weighted_Hq_for_answer)
      Fk_b = f.tanh(attended_match_lstm_b + self.attend_answer(hb) + weighted_Hq_for_answer)

      # beta_k[_b]_scores.shape = (seq_len, batch, 1)
      beta_ks = []
      beta_k_scores = self.beta_transform(Fk)
      beta_k_bs = []
      beta_k_b_scores = self.beta_transform(Fk_b)
      for idx in range(batch_size):
        beta_k_idx = f.softmax(beta_k_scores[:passage_lens[idx],idx,:], dim=0)
        beta_k_b_idx = f.softmax(beta_k_b_scores[:passage_lens[idx],idx,:], dim=0)

        if beta_k_idx.size()[0] < max_passage_len:
          diff = max_passage_len - beta_k_idx.size()[0]
          zeros = self.variable(torch.zeros((diff, 1)))
          beta_k_idx = torch.cat((beta_k_idx, zeros), dim=0)
          beta_k_b_idx = torch.cat((beta_k_b_idx, zeros), dim=0)

        # beta_k[_b]_idx.shape = (max_seq_len, 1)
        beta_ks.append(beta_k_idx)
        beta_k_bs.append(beta_k_b_idx)

        if k > 0:
					t_k = k-1
					mle_losses.append(-torch.log(torch.squeeze(beta_k_idx[answer[t_k][idx]])))
					mle_losses.append(-torch.log(torch.squeeze(beta_k_b_idx[answer[1-t_k][idx]])))

      # beta_k.shape = (seq_len, batch, 1)
      beta_k = torch.stack(beta_ks, dim=1)
      beta_k_b = torch.stack(beta_k_bs, dim=1)

      # Store distribution over passage words for answer start/end.
      if(k > 0):
        answer_distributions.append(torch.t(torch.squeeze(beta_k, dim=-1)))
        answer_distributions_b.append(torch.t(torch.squeeze(beta_k_b, dim=-1)))

      # weighted_Hr.shape = (batch, 2*hdim + num_pos_tags)
      weighted_Hr = torch.squeeze(torch.bmm(beta_k.permute(1, 2, 0),
                                            torch.transpose(Hr, 0, 1)), dim=1)
      weighted_Hr_b = torch.squeeze(torch.bmm(beta_k_b.permute(1, 2, 0),
                                              torch.transpose(Hr, 0, 1)), dim=1)

      # a{f,b}.shape = (batch, 6*hdim + 2*num_pos_tags)
      af = torch.cat((weighted_Hr, weighted_Hq_for_answer), dim=-1)
      ab = torch.cat((weighted_Hr_b, weighted_Hq_for_answer), dim=-1)

      # LSTM step.
      ha, ca = self.answer_pointer_lstm(af, (ha, ca))
      hb, cb = self.answer_pointer_lstm(ab, (hb, cb))

    # Compute the loss.
    loss_f = -torch.log(
                (torch.bmm(torch.unsqueeze(answer_distributions[0], -1),
                          torch.unsqueeze(answer_distributions[1], 1)) * \
		             f1_matrices).view(batch_size, -1).sum(1)).sum()
    loss_b = -torch.log(
                (torch.bmm(torch.unsqueeze(answer_distributions_b[1], -1),
                           torch.unsqueeze(answer_distributions_b[0], 1)) * \
		             f1_matrices).view(batch_size, -1).sum(1)).sum()
    loss = self.f1_loss_ratio * (loss_f + loss_b) + \
           (1 - self.f1_loss_ratio) * sum(mle_losses)
    loss /= batch_size
    return answer_distributions, answer_distributions_b, loss

  # Forward pass method.
  # passage = tuple((seq_len, batch), len_within_batch)
  # question = tuple((seq_len, batch), len_within_batch)
  # answer = tuple((2, batch))
  # f1_matrices = (batch, seq_len, seq_len)
  # question_pos_tags = (seq_len, batch, num_pos_tags)
  # passage_pos_tags = (seq_len, batch, num_pos_tags)
  def forward(self, passage, question, answer, f1_matrices, question_pos_tags,
              passage_pos_tags):
    if not self.use_glove:
      padded_passage = self.placeholder(passage[0], False)
      padded_question = self.placeholder(question[0], False)
    batch_size = passage[0].shape[1]
    max_passage_len = passage[0].shape[0]
    max_question_len = question[0].shape[0]
    passage_lens = passage[1]
    question_lens = question[1]
    f1_mat = self.placeholder(f1_matrices)

    # Get embedded passage and question representations.
    if not self.use_glove:
      p = torch.transpose(self.embedding(torch.t(padded_passage)), 0, 1)
      q = torch.transpose(self.embedding(torch.t(padded_question)), 0, 1)
    else:
      p = self.get_glove_embeddings(passage[0])
      q = self.get_glove_embeddings(question[0])

    # Embedding input dropout.
    # {p,q}.shape = (seq_len, batch, embedding_dim + num_pos_tags)
    p = torch.cat((p, self.placeholder(passage_pos_tags)), dim=-1)
    q = torch.cat((q, self.placeholder(question_pos_tags)), dim=-1)

    # Preprocessing LSTM outputs for passage and question input.
    # H{p,q}.shape = (seq_len, batch, 2 * hdim)
    Hp = p
    for layer_no in range(self.num_preprocessing_layers):
      Hp = getattr(self, 'dropoutp_' + str(layer_no))(Hp)
      Hp = self.preprocess_input(str(layer_no), Hp, max_passage_len, passage_lens,
                                 batch_size)

    Hq = q
    for layer_no in range(self.num_preprocessing_layers):
      Hq = getattr(self, 'dropoutq_' + str(layer_no))(Hq)
      Hq = self.preprocess_input(str(layer_no), Hq, max_question_len, question_lens,
                                 batch_size)

    # Bi-directional multi-layer MatchLSTM for passage-aware question representation.
    # Hq_aware.shape = (seq_len, batch, 2 * hdim)
    Hq_aware = Hq
    for layer_no in range(self.num_question_matchlstm_layers):
      Hq_aware = self.match_passage_question(str(layer_no), Hq_aware, Hp,
                                             max_question_len, question_lens,
                                             batch_size)
      # Passage-aware question representation dropout.
      Hq_aware = getattr(self, 'dropout_question_matchlstm_' + str(layer_no))(Hq_aware)

    # Hq.shape = (seq_len, batch, 4 * hdim + num_pos_tags)
    Hq = torch.cat((Hq, Hq_aware, self.placeholder(question_pos_tags)), dim=-1)

    # Bi-directional multi-layer MatchLSTM for question-aware passage representation.
    Hr = Hp
    for layer_no in range(self.num_passage_matchlstm_layers):
      Hr = self.match_question_passage(str(layer_no), Hr, Hq, max_passage_len,
                                       passage_lens, batch_size)
      # For the last layer output, also concatenate passage POS tags.
      if layer_no == self.num_passage_matchlstm_layers - 1:
        Hr = torch.cat((Hr, self.placeholder(passage_pos_tags)), dim=-1)
      # Question-aware passage representation dropout.
      Hr = getattr(self, 'dropout_passage_matchlstm_' + str(layer_no))(Hr)

    # Get probability distributions over the answer start, answer end,
    # and the loss for training.
    # At this point, Hr.shape = (seq_len, batch, 2 * hdim + num_pos_tags)
    answer_distributions, answer_distributions_b, loss = \
      self.answer_pointer(Hr, Hp, Hq, max_question_len, question_lens,
                          max_passage_len, passage_lens, batch_size, answer,
                          f1_mat)

    self.loss = loss
    return answer_distributions, answer_distributions_b

