import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class qNet(nn.Module):
  ''' Match-LSTM model definition. Properties specified in config.'''

  # Constructor
  def __init__(self, config, debug = False):
    # Call constructor of nn module.
    super(qNet, self).__init__()

    # Set-up parameters from config.
    self.load_from_config(config)

    # Construct the model, storing all necessary layers.
    self.build_model(debug)
    self.debug = debug

  # Load configuration options
  def load_from_config(self, config):
    self.embed_size = config['embed_size']
    self.vocab_size = config['vocab_size']
    self.hidden_size = config['hidden_size']
    self.attention_size = config['attention_size']
    self.lr_rate = config['lr']
    self.glove_path = config['glove_path']
    self.optimizer = config['optimizer']
    self.index_to_word = config['index_to_word']
    self.word_to_index = config['word_to_index']
    self.use_glove = config['use_glove']
    self.use_cuda = config['cuda']
    self.dropout = config['dropout']
    self.f1_loss_multiplier = config['f1_loss_multiplier']
    self.f1_loss_threshold = config['f1_loss_threshold']
    self.num_pos_tags = config['num_pos_tags']
    self.num_ner_tags = config['num_ner_tags']
    self.num_preprocessing_layers = config['num_preprocessing_layers']
    self.num_postprocessing_layers = config['num_postprocessing_layers']
    self.num_matchlstm_layers = config['num_matchlstm_layers']
    self.num_selfmatch_layers = config['num_selfmatch_layers']

  def build_model(self, debug):
    # Embedding look-up.
    self.oov_count = 0
    self.oov_list = []
    if self.use_glove and not debug:
      embeddings = np.zeros((self.vocab_size, self.embed_size))
      with open(self.glove_path) as f:
        for line in f:
          word = line[:line.index(" ")]
          if not word in self.word_to_index:
            continue
          line = line.split()
          embeddings[self.word_to_index[line[0]]] = np.array(map(float,line[1:]))
      for i, embedding in enumerate(embeddings):
        if sum(embedding) == 0:
          self.oov_count += 1
          self.oov_list.append(self.index_to_word[i])
      self.embedding = embeddings
    elif debug:
      self.embedding = np.zeros((self.vocab_size, self.embed_size))
    else:
      self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                    self.word_to_index['<pad>'])

    # Input dropouts before pre-processing.
    self.dropout_p = nn.Dropout(self.dropout)
    self.dropout_q = nn.Dropout(self.dropout)

    # Passage and Question pre-processing LSTMs (matrices Hp and Hq respectively).
    self.preprocessing_lstm = \
      nn.LSTM(input_size = self.embed_size + self.num_pos_tags + self.num_ner_tags,
              hidden_size = self.hidden_size // 2,
              num_layers = self.num_preprocessing_layers,
              dropout = self.dropout,
              bidirectional = True)

    # Tie forward and backward pre-processing LSTM weights.
    for weight_type in ["hh", "ih"]:
      for layer_no in range(self.num_preprocessing_layers):
        weight_name = "weight_" + weight_type + "_l" + str(layer_no)
        bias_name = "bias_" + weight_type + "_l" + str(layer_no)
        setattr(self.preprocessing_lstm, weight_name + "_reverse",
                getattr(self.preprocessing_lstm, weight_name))
        setattr(self.preprocessing_lstm, bias_name + "_reverse",
                getattr(self.preprocessing_lstm, bias_name))

    # Attention transformations (variable names below given against those in
    # Wang, Shuohang, and Jing Jiang. "Machine comprehension using match-lstm
    # and answer pointer." arXiv preprint arXiv:1608.07905 (2016).)
    for layer_no in range(self.num_matchlstm_layers):
      setattr(self, 'attend_question_for_passage_' + str(layer_no),
              nn.Linear(self.hidden_size, self.attention_size,
                        bias = False))
      setattr(self, 'attend_passage_for_passage_' + str(layer_no),
              nn.Linear(self.hidden_size, self.attention_size))
      setattr(self, 'attend_passage_hidden_' + str(layer_no),
              nn.Linear(self.hidden_size // 2, self.attention_size, bias = False))
      setattr(self, 'passage_alpha_transform_' + str(layer_no),
              nn.Linear(self.attention_size, 1))
      # Final Match-LSTM cells (bi-directional).
      setattr(self, 'passage_match_lstm_' + str(layer_no),
              nn.LSTMCell(input_size = self.hidden_size * 2,
                          hidden_size = self.hidden_size // 2))
      setattr(self, 'dropout_passage_matchlstm_' + str(layer_no),
              nn.Dropout(self.dropout))

    # Passage self-matching layers.
    for layer_no in range(self.num_selfmatch_layers):
      setattr(self, 'attend_self_passage_' + str(layer_no),
              nn.Linear(self.hidden_size, self.attention_size, bias = False))
      setattr(self, 'attend_self_hidden_' + str(layer_no),
              nn.Linear(self.hidden_size // 2, self.attention_size, bias = False))
      setattr(self, 'self_alpha_transform_' + str(layer_no),
              nn.Linear(self.attention_size, 1))
      # Final Self-matching LSTM cells (bi-directional).
      setattr(self, 'self_match_lstm_' + str(layer_no),
              nn.LSTMCell(input_size = self.hidden_size * 2,
                          hidden_size = self.hidden_size // 2))
      setattr(self, 'dropout_self_matchlstm_' + str(layer_no),
              nn.Dropout(self.dropout))

    # Question-aware passage post-processing LSTM.
    if self.num_postprocessing_layers > 0:
      self.postprocessing_lstm = nn.LSTM(input_size = self.hidden_size,
                                         hidden_size = self.hidden_size // 2,
                                         num_layers = self.num_postprocessing_layers,
                                         dropout = self.dropout,
                                         bidirectional = True)

    # Tie forward and backward post-processing LSTM weights.
    for weight_type in ["hh", "ih"]:
      for layer_no in range(self.num_postprocessing_layers):
        weight_name = "weight_" + weight_type + "_l" + str(layer_no)
        bias_name = "bias_" + weight_type + "_l" + str(layer_no)
        setattr(self.postprocessing_lstm, weight_name + "_reverse",
                getattr(self.postprocessing_lstm, weight_name))
        setattr(self.postprocessing_lstm, bias_name + "_reverse",
                getattr(self.postprocessing_lstm, bias_name))

    # Answer pointer attention transformations.
    # Question attentions for answer sentence pointer network.
    setattr(self, 'attend_question',
            nn.Linear(self.hidden_size, self.attention_size))
    setattr(self, 'alpha_transform',
            nn.Linear(self.attention_size, 1))

    # Attend to the input.
    setattr(self, 'attend_input',
            nn.Linear(self.hidden_size, self.attention_size, bias = False))
    # Attend to answer hidden state.
    setattr(self, 'attend_answer',
            nn.Linear(self.hidden_size // 2, self.attention_size, bias = False))

    setattr(self, 'beta_transform',
            nn.Linear(self.attention_size, 1))

    # Answer pointer LSTM.
    setattr(self, 'answer_pointer_lstm',
            nn.LSTMCell(input_size = self.hidden_size * 2,
                        hidden_size = self.hidden_size // 2))

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
  def get_initial_lstm(self, batch_size, hidden_size = None, for_cell = True):
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
    return self.placeholder(self.embedding[inp])

  # Get final layer hidden states of the provided LSTM run over the given
  # input sequence.
  def process_input_with_lstm(self, inputs, max_len, input_lens, batch_size,
                              lstm_to_use):
    idxs = np.array(np.argsort(input_lens)[::-1])
    lens = [ input_lens[idx] for idx in idxs ]
    idxs = self.variable(torch.from_numpy(idxs))
    inputs_sorted = torch.index_select(inputs, 1, idxs)
    inputs_sorted = pack_padded_sequence(inputs_sorted, lens)
    H, _ = lstm_to_use(inputs_sorted)
    H, _ = pad_packed_sequence(H)
    unsorted_idxs = self.variable(torch.zeros(idxs.size()[0])).long()
    unsorted_idxs.scatter_(0, idxs,
                           self.variable(torch.arange(idxs.size()[0])).long())
    return torch.index_select(H, 1, unsorted_idxs)

  # Get a question-aware passage representation.
  def match_question_passage(self, layer_no, Hpi, Hq, max_passage_len,
                             passage_lens, batch_size, mask, mask_q_byte,
                             mask_p_zero, mask_q_zero):
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (batch, hdim / 2)
    hf, cf = self.get_initial_lstm(batch_size, self.hidden_size // 2)
    hb, cb = self.get_initial_lstm(batch_size, self.hidden_size // 2)

    # Get vectors zi for each i in passage.
    # Attended question is the same at each time step. Just compute it once.
    # attended_question.shape = (seq_len, batch, hdim)
    attended_question = getattr(self, 'attend_question_for_passage_' + layer_no)(Hq)
    attended_passage = getattr(self, 'attend_passage_for_passage_' + layer_no)(Hpi)
    attention_q_plus_p = []
    for t in range(max_passage_len):
      attention_q_plus_p.append(attended_question + \
                                (attended_passage[t] * mask[t]))
    transposed_Hq = torch.transpose(Hq, 0, 1)
    Hf, Hb = [], []
    for i in range(max_passage_len):
        forward_idx = i
        backward_idx = max_passage_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        gf = f.tanh(attention_q_plus_p[forward_idx] + \
                 getattr(self, 'attend_passage_hidden_' + layer_no)(hf))
        gb = f.tanh(attention_q_plus_p[backward_idx] + \
                 getattr(self, 'attend_passage_hidden_' + layer_no)(hb))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = getattr(self, 'passage_alpha_transform_' + layer_no)(gf)
        alpha_b = getattr(self, 'passage_alpha_transform_' + layer_no)(gb)

        # Mask out padded regions of question attention in the batch.
        # -inf ensures that the post-softmax output for those parts of the
        # output is zero.
        alpha_f.masked_fill_(mask_q_byte, -float('inf'))
        alpha_b.masked_fill_(mask_q_byte, -float('inf'))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = f.softmax(alpha_f, dim=0) * mask_q_zero
        alpha_b = f.softmax(alpha_b, dim=0) * mask_q_zero

        # Hp[{forward,backward}_idx].shape = (batch, hdim)
        # Hq = (seq_len, batch, hdim)
        # weighted_Hq_f.shape = (batch, hdim)
        weighted_Hq_f = torch.squeeze(torch.bmm(alpha_f.permute(1, 2, 0),
                                      transposed_Hq), dim=1)
        weighted_Hq_b = torch.squeeze(torch.bmm(alpha_b.permute(1, 2, 0),
                                      transposed_Hq), dim=1)

        # z{f,b}.shape = (batch, 2 * hdim)
        zf = torch.cat((Hpi[forward_idx], weighted_Hq_f), dim=-1)
        zb = torch.cat((Hpi[backward_idx], weighted_Hq_b), dim=-1)

        # Take forward and backward LSTM steps, with zf and zb as inputs.
        hf, cf = getattr(self, 'passage_match_lstm_' + layer_no)(zf, (hf, cf))
        hb, cb = getattr(self, 'passage_match_lstm_' + layer_no)(zb, (hb, cb))

        # Back to initial zero states for padded regions.
        hf = hf * mask[forward_idx]
        cf = cf * mask[forward_idx]
        hb = hb * mask[backward_idx]
        cb = cb * mask[backward_idx]

        # Append hidden states to create Hf and Hb matrices.
        # h{f,b}.shape = (batch, hdim / 2)
        Hf.append(hf)
        Hb.append(hb)

    # H{f,b}.shape = (seq_len, batch, hdim / 2)
    Hb = Hb[::-1]
    Hf = torch.stack(Hf, dim=0)
    Hb = torch.stack(Hb, dim=0)

    # Hr.shape = (seq_len, batch, hdim)
    Hr = torch.cat((Hf, Hb), dim=-1)
    return Hr

  # Get a self-aware (question-aware) passage representation.
  def match_passage_passage(self, layer_no, Hr, max_passage_len, passage_lens,
                            batch_size, mask_p_byte, mask_p, mask_p_zero,
                            mask_q_zero):
    # Initial hidden and cell states for forward and backward LSTMs.
    # h{f,b}.shape = (batch, hdim / 2)
    hf, cf = self.get_initial_lstm(batch_size, self.hidden_size // 2)
    hb, cb = self.get_initial_lstm(batch_size, self.hidden_size // 2)

    # Get vectors zi for each i in passage.
    # Attended passage is the same at each time step. Just compute it once.
    # attended_passage.shape = (seq_len, batch, hdim)
    attended_passage = getattr(self, 'attend_self_passage_' + layer_no)(Hr)
    transposed_Hr = torch.transpose(Hr, 0, 1)
    Hf, Hb = [], []
    for i in range(max_passage_len):
        forward_idx = i
        backward_idx = max_passage_len-i-1
        # g{f,b}.shape = (seq_len, batch, hdim)
        gf = f.tanh(attended_passage + \
                 getattr(self, 'attend_self_hidden_' + layer_no)(hf))
        gb = f.tanh(attended_passage + \
                 getattr(self, 'attend_self_hidden_' + layer_no)(hb))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = getattr(self, 'self_alpha_transform_' + layer_no)(gf)
        alpha_b = getattr(self, 'self_alpha_transform_' + layer_no)(gb)

        # Mask out padded regions of passage attention in the batch.
        # -inf ensures that the post-softmax output for those parts of the
        # output is zero.
        alpha_f.masked_fill_(mask_p_byte, -float('inf'))
        alpha_b.masked_fill_(mask_p_byte, -float('inf'))

        # alpha_{f,g}.shape = (seq_len, batch, 1)
        alpha_f = f.softmax(alpha_f, dim=0) * mask_p_zero
        alpha_b = f.softmax(alpha_b, dim=0) * mask_p_zero

        # Hr[{forward,backward}_idx].shape = (batch, hdim)
        # weighted_Hr_{f,b}.shape = (batch, hdim)
        weighted_Hr_f = torch.squeeze(torch.bmm(alpha_f.permute(1, 2, 0),
                                      transposed_Hr), dim=1)
        weighted_Hr_b = torch.squeeze(torch.bmm(alpha_b.permute(1, 2, 0),
                                      transposed_Hr), dim=1)

        # z{f,b}.shape = (batch, 2 * hdim)
        zf = torch.cat((Hr[forward_idx], weighted_Hr_f), dim=-1)
        zb = torch.cat((Hr[backward_idx], weighted_Hr_b), dim=-1)

        # Take forward and backward LSTM steps, with zf and zb as inputs.
        hf, cf = getattr(self, 'self_match_lstm_' + layer_no)(zf, (hf, cf))
        hb, cb = getattr(self, 'self_match_lstm_' + layer_no)(zb, (hb, cb))

        # Back to initial zero states for padded regions.
        hf = hf * mask_p[forward_idx]
        cf = cf * mask_p[forward_idx]
        hb = hb * mask_p[backward_idx]
        cb = cb * mask_p[backward_idx]

        # Append hidden states to create Hf and Hb matrices.
        # h{f,b}.shape = (batch, hdim / 2)
        Hf.append(hf)
        Hb.append(hb)

    # H{f,b}.shape = (seq_len, batch, hdim / 2)
    Hb = Hb[::-1]
    Hf = torch.stack(Hf, dim=0)
    Hb = torch.stack(Hb, dim=0)

    # Hr.shape = (seq_len, batch, hdim)
    Hr = torch.cat((Hf, Hb), dim=-1)
    return Hr

  # Boundary pointer model, that gives probability distributions over the
  # start and end indices. Returns the hidden states, as well as the predicted
  # distributions.
  def answer_pointer(self, Hr, Hp, Hq, max_question_len, question_lens,
                     max_passage_len, passage_lens, batch_size, mask_p_byte,
                     mask_q_byte, mask_p_zero, mask_q_zero):
    # attended_input[_b].shape = (seq_len, batch, hdim)
    attended_input = getattr(self, 'attend_input')(Hr)

    # weighted_Hq.shape = (batch, hdim)
    attended_question = f.tanh(getattr(self, 'attend_question')(Hq))
    alpha_q = getattr(self, 'alpha_transform')(attended_question)
    alpha_q.masked_fill_(mask_q_byte, -float('inf'))
    alpha_q = f.softmax(alpha_q, dim=0) * mask_q_zero
    weighted_Hq = torch.squeeze(torch.bmm(alpha_q.permute(1, 2, 0),
                                          torch.transpose(Hq, 0, 1)), dim=1)

    # {h,c}{a,b}.shape = (batch, hdim / 2)
    ha, ca = self.get_initial_lstm(batch_size, self.hidden_size // 2)
    hb, cb = self.get_initial_lstm(batch_size, self.hidden_size // 2)

    answer_distributions = []
    answer_distributions_b = []

    # Two three-step LSTMs:
    #   1) Point to the start index first, then the end index.
    #   2) Point to the end index first, then the start index.
    # 1st step initializes the hidden states to some answer representations.
    # 2nd step predicts start/end distributions in 1/2 respectively.
    # 3rd step predicts end/start distributions in 1/2 respectively.
    for k in range(3):
      # Fk[_b].shape = (seq_len, batch, hdim)
      Fk = f.tanh(attended_input + \
                  getattr(self, 'attend_answer')(ha))
      Fk_b = f.tanh(attended_input + \
                    getattr(self, 'attend_answer')(hb))

      # Get softmaxes over only valid paragraph lengths for each element in
      # the batch.
      # beta_k[_b].shape = (seq_len, batch, 1)
      beta_k = getattr(self, 'beta_transform')(Fk)
      beta_k_b = getattr(self, 'beta_transform')(Fk_b)

      # Mask out padded regions.
      beta_k.masked_fill_(mask_p_byte, -float('inf'))
      beta_k_b.masked_fill_(mask_p_byte, -float('inf'))

      beta_k = f.softmax(beta_k, dim=0) * mask_p_zero
      beta_k_b = f.softmax(beta_k_b, dim=0) * mask_p_zero

      # Store distributions produced at start and end prediction steps.
      if k > 0:
        answer_distributions.append(torch.t(torch.squeeze(beta_k, dim=-1)))
        answer_distributions_b.append(torch.t(torch.squeeze(beta_k_b, dim=-1)))

      # Only the first two steps of the answer pointer are useful beyond
      # this point.
      if k >= 2:
        break

      # weighted_Hr.shape = (batch, hdim)
      weighted_Hr = torch.squeeze(torch.bmm(beta_k.permute(1, 2, 0),
                                            torch.transpose(Hr, 0, 1)), dim=1)
      weighted_Hr_b = torch.squeeze(torch.bmm(beta_k_b.permute(1, 2, 0),
                                              torch.transpose(Hr, 0, 1)), dim=1)

      # a{f,b}.shape = (batch, 2 * hdim)
      af = torch.cat((weighted_Hr, weighted_Hq), dim=-1)
      ab = torch.cat((weighted_Hr_b, weighted_Hq), dim=-1)

      # LSTM step.
      ha, ca = getattr(self, 'answer_pointer_lstm')(af, (ha, ca))
      hb, cb = getattr(self, 'answer_pointer_lstm')(ab, (hb, cb))

    return answer_distributions, answer_distributions_b

  # Boundary pointer model, that gives probability distributions over the
  # answer start and answer end indices. Additionally returns the loss
  # for training.
  def point_at_answer(self, Hr, Hp, Hq, max_question_len, question_lens,
                      max_passage_len, passage_lens, batch_size,
                      answer, f1_matrices, mask_p_byte, mask_q_byte,
                      mask_p_zero, mask_q_zero):
    # Predict the answer start and end indices.
    distribution = self.answer_pointer(Hr, Hp, Hq, max_question_len,
                                       question_lens, max_passage_len,
                                       passage_lens, batch_size, mask_p_byte,
                                       mask_q_byte, mask_p_zero, mask_q_zero)

    batch_losses = [ [] for _ in range(batch_size) ]
    # For each example in the batch, add the negative log of answer start
    # and end index probabilities to the MLE loss, from both forward and
    # backward answer pointers.
    for idx in range(batch_size):
      batch_losses[idx].append(
        distribution[0][0][idx, answer[0][idx]] *\
        distribution[0][1][idx, answer[1][idx]] *\
        distribution[1][0][idx, answer[1][idx]] *\
        distribution[1][1][idx, answer[0][idx]])
    if self.f1_loss_multiplier > 0:
      # Compute the F1 distribution loss.
      # f1_matrices.shape = (batch, max_seq_len, max_seq_len)
      # If there is thresholding, only penalize values below that threshold.
      if self.f1_loss_threshold >= 0:
        f1_matrices = self.placeholder(f1_matrices > self.f1_loss_threshold,
                                       to_float = True)
      else:
        f1_matrices = self.placeholder(f1_matrices)
      loss_f1_f = (torch.bmm(torch.unsqueeze(distribution[0][0], -1),
                             torch.unsqueeze(distribution[0][1], 1)) * \
                   f1_matrices).view(batch_size, -1).sum(1)
      loss_f1_b = (torch.bmm(torch.unsqueeze(distribution[1][1], -1),
                             torch.unsqueeze(distribution[1][0], 1)) * \
                   f1_matrices).view(batch_size, -1).sum(1)
      for idx in range(batch_size):
        batch_losses[idx].append(self.f1_loss_multiplier * loss_f1_f[idx] *\
                                 loss_f1_b[idx])

    loss = 0.0
    for idx in range(batch_size):
      loss += -torch.log(sum(batch_losses[idx]) / (1 + self.f1_loss_multiplier))
    loss /= batch_size
    return distribution, loss

  # Get matrix for padding hidden states of an LSTM running over the
  # given maximum length, for lengths in the batch.
  def get_mask_matrix(self, batch_size, max_len, lens):
    mask_matrix = []
    for t in range(max_len):
      mask = np.array([ [1.0] if t < lens[i] else [0.0] \
                          for i in range(batch_size) ])
      mask = self.placeholder(mask)
      mask_matrix.append(mask)
    return mask_matrix

  # Forward pass method.
  # passage = tuple((seq_len, batch), len_within_batch)
  # question = tuple((seq_len, batch), len_within_batch)
  # answer = tuple((2, batch))
  # f1_matrices = (batch, seq_len, seq_len)
  # question_pos_tags = (seq_len, batch, num_pos_tags)
  # question_ner_tags = (seq_len, batch, num_ner_tags)
  # passage_pos_tags = (seq_len, batch, num_pos_tags)
  # passage_ner_tags = (seq_len, batch, num_ner_tags)
  # answer_sentence = ((2, batch))
  def forward(self, passage, question, answer, f1_matrices,
              question_pos_tags, question_ner_tags, passage_pos_tags,
              passage_ner_tags, answer_sentence):
    if not self.use_glove:
      padded_passage = self.placeholder(passage[0], False)
      padded_question = self.placeholder(question[0], False)
    batch_size = passage[0].shape[1]
    max_passage_len = passage[0].shape[0]
    max_question_len = question[0].shape[0]
    passage_lens = passage[1]
    question_lens = question[1]

    if self.debug:
      start_prepare = time.time()

    mask_p = self.get_mask_matrix(batch_size, max_passage_len, passage_lens)
    mask_q = self.get_mask_matrix(batch_size, max_question_len, question_lens)

    # mask_q.shape = (seq_len, batch, 1)
    mask_p_byte = (1-torch.stack(mask_p, dim=0)).byte()
    mask_q_byte = (1-torch.stack(mask_q, dim=0)).byte()
    mask_p_zero = torch.stack(mask_p, dim=0).float()
    mask_q_zero = torch.stack(mask_q, dim=0).float()

    # Get embedded passage and question representations.
    if not self.use_glove:
      p = torch.transpose(self.embedding(torch.t(padded_passage)), 0, 1)
      q = torch.transpose(self.embedding(torch.t(padded_question)), 0, 1)
    else:
      p = self.get_glove_embeddings(passage[0])
      q = self.get_glove_embeddings(question[0])

    # Embedding input dropout.
    # {p,q}.shape = (seq_len, batch, embedding_dim + num_pos_tags + num_ner_tags)
    p = torch.cat((p, self.placeholder(passage_pos_tags),
                   self.placeholder(passage_ner_tags)), dim=-1)
    q = torch.cat((q, self.placeholder(question_pos_tags),
                   self.placeholder(question_ner_tags)), dim=-1)

    # Apply input dropouts.
    p = self.dropout_p(p)
    q = self.dropout_q(q)

    if self.debug:
      p.sum()
      q.sum()
      print "Data preparation time: %.2fs" % (time.time() - start_prepare)
      start_preprocess = time.time()

    # Preprocessing LSTM outputs for passage and question input.
    # H{p,q}.shape = (seq_len, batch, hdim)
    Hp = self.process_input_with_lstm(p, max_passage_len, passage_lens, batch_size,
                                      self.preprocessing_lstm)
    Hq = self.process_input_with_lstm(q, max_question_len, question_lens, batch_size,
                                      self.preprocessing_lstm)

    if self.debug:
      Hp.sum()
      Hq.sum()
      print "Data pre-processing time: %.2fs" % (time.time() - start_preprocess)
      start_matching = time.time()

    # Bi-directional multi-layer MatchLSTM for question-aware passage representation.
    Hr = Hp
    for layer_no in range(self.num_matchlstm_layers):
      Hr = self.match_question_passage(str(layer_no), Hr, Hq, max_passage_len,
                                       passage_lens, batch_size, mask_p, mask_q_byte,
                                       mask_p_zero, mask_q_zero)
      # Question-aware passage representation dropout.
      Hr = getattr(self, 'dropout_passage_matchlstm_' + str(layer_no))(Hr)

    if self.debug:
      Hr.sum()
      print "Matching passage with question time: %.2fs" % \
            (time.time() - start_matching)
      start_postprocess = time.time()

    # (Question-aware) passage self-matching layers.
    for layer_no in range(self.num_selfmatch_layers):
      Hr = self.match_passage_passage(str(layer_no), Hr, max_passage_len,
                                      passage_lens, batch_size, mask_p_byte, mask_p,
                                      mask_p_zero, mask_q_zero)
      # Passage self-matching layer dropout.
      Hr = getattr(self, 'dropout_self_matchlstm_' + str(layer_no))(Hr)

    if self.debug and self.num_selfmatch_layers > 0:
      Hr.sum()
      print "Self-matching time: %.2fs" % (time.time() - start_postprocess)
      start_postprocess = time.time()

    if self.num_postprocessing_layers > 0:
      Hr = self.process_input_with_lstm(Hr, max_passage_len, passage_lens, batch_size,
                                        self.postprocessing_lstm)

    if self.debug:
      Hr.sum()
      print "Post-processing question-aware passage time: %.2fs" % \
            (time.time() - start_postprocess)
      start_answer = time.time()

    # Get probability distributions over the answer start, answer end,
    # and the loss for training.
    # At this point, Hr.shape = (seq_len, batch, hdim)
    answer_distributions_list, loss = \
      self.point_at_answer(Hr, Hp, Hq, max_question_len, question_lens,
                           max_passage_len, passage_lens, batch_size,
                           answer, f1_matrices, mask_p_byte, mask_q_byte,
                           mask_p_zero, mask_q_zero)

    if self.debug:
      loss.data[0]
      print "Answer pointer time: %.2fs" % (time.time() - start_answer)

    self.loss = loss
    return answer_distributions_list

