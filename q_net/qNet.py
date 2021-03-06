import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class qNet(nn.Module):
  ''' Q-NET model definition. Properties specified in config.'''

  # Constructor
  def __init__(self, config, debug_level = 0):
    # Call constructor of nn module.
    super(qNet, self).__init__()

    # Set-up parameters from config.
    self.load_from_config(config)

    # Construct the model, storing all necessary layers.
    self.build_model(debug_level)
    self.debug_level = debug_level

    # Volatile variables for inference. If true, computation graph isn't built.
    self.volatile = False

  # Load configuration options
  def load_from_config(self, config):
    self.embed_size = config['embed_size']
    self.vocab_size = config['vocab_size']
    self.hidden_size = config['hidden_size']
    self.attention_size = config['attention_size']
    self.lr_rate = config['lr']
    self.vectors_path = config['vectors_path']
    self.optimizer = config['optimizer']
    self.index_to_word = config['index_to_word']
    self.word_to_index = config['word_to_index']
    self.use_pretrained = config['use_pretrained']
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

  def load_embeddings(self, debug_level):
    # Embedding look-up.
    self.oov_count = 0
    self.oov_list = []
    known_idxs, unknown_idxs = [], []
    if self.use_pretrained and debug_level <= 1:
      # Read embeddings from file.
      embeddings = np.zeros((self.vocab_size, self.embed_size))
      with open(self.vectors_path) as f:
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
          unknown_idxs.append(i)
        else:
          known_idxs.append(i)

      # Initialize unknown word embeddings by sampling from multivariate normal distribution,
      # with mean and covariance as sample mean and covariance of known embeddings.
      known_embeddings = embeddings[known_idxs]
      np_rng = np.random.RandomState(123)
      known_mean = np.mean(known_embeddings, axis=0)
      known_covar = np.cov(known_embeddings, rowvar=0)
      unknown_embeddings = np_rng.multivariate_normal(mean=known_mean, cov=known_covar,
                                                      size=len(unknown_idxs)).astype(np.float32)
      embeddings[unknown_idxs] = unknown_embeddings
      self.embedding = embeddings
    elif debug_level >= 2:
      # Initialize all embeddings with zero for debugging.
      self.embedding = np.zeros((self.vocab_size, self.embed_size))
    else:
      # Create trainable embeddings layer.
      self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                    self.word_to_index['<pad>'])

  def build_model(self, debug_level):
    # Read embeddings from file, create all zeros for debug, or make a trainable layer.
    self.load_embeddings(debug_level)

    # Passage and Question pre-processing LSTMs (matrices Hp and Hq respectively).
    self.preprocessing_lstm = \
      nn.LSTM(input_size = self.embed_size + self.num_pos_tags + self.num_ner_tags,
              hidden_size = self.hidden_size // 2,
              num_layers = self.num_preprocessing_layers,
              dropout = self.dropout,
              bidirectional = True)

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

    # Answer pointer attention transformations.
    # Question attentions for answer sentence pointer network.
    setattr(self, 'attend_question',
            nn.Linear(self.hidden_size, self.attention_size))
    setattr(self, 'alpha_transform',
            nn.Linear(self.attention_size, 1))

    # Attend to the input.
    setattr(self, 'attend_input',
            nn.Linear(self.hidden_size, self.attention_size))
    setattr(self, 'attend_input_b',
            nn.Linear(self.hidden_size, self.attention_size))
    # Attend to answer hidden state.
    setattr(self, 'attend_answer',
            nn.Linear(self.hidden_size // 2, self.attention_size,
                      bias = False))

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

  def set_train(self):
    self.volatile = False
    self.train()

  def set_eval(self):
    self.volatile = True
    self.eval()

  def free_memory(self):
    del self.loss
    del self.mle_loss
    if self.f1_loss_multiplier > 0:
      del self.f1_loss

  def load_from_file(self, path):
    self = torch.load(path)
    return self

  def variable(self, v):
    if self.use_cuda:
      return Variable(v, requires_grad = False, volatile = self.volatile).cuda()
    return Variable(v, requires_grad = False, volatile = self.volatile)

  def placeholder(self, np_var, to_float=True):
    if to_float:
      np_var = np_var.astype(np.float32)
    v = self.variable(torch.from_numpy(np_var))
    return v

  # Get an initial tuple of (h0, c0).
  # h0, c0 have dims (num_directions * num_layers, batch_size, hidden_size)
  # If for a cell, they have dims (batch_size, hidden_size)
  def get_initial_lstm(self, batch_size, hidden_size = None, for_cell = True):
    tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
    if hidden_size is None:
      hidden_size = self.hidden_size
    if not for_cell:
      return (Variable(tensor(1, batch_size, hidden_size).zero_(),
                       requires_grad = False,
                       volatile = self.volatile),
              Variable(tensor(1, batch_size, hidden_size).zero_(),
                       requires_grad = False,
                       volatile = self.volatile))
    return (Variable(tensor(batch_size, hidden_size).zero_(),
                     volatile = self.volatile,
                     requires_grad = False),
            Variable(tensor(batch_size, hidden_size).zero_(),
                     volatile = self.volatile,
                     requires_grad = False))

  # inp.shape = (seq_len, batch)
  # output.shape = (seq_len, batch, embed_size)
  def get_vector_embeddings(self, inp):
    return self.placeholder(self.embedding[inp])

  # Detach input of (seq_len, batch, ...) for the given indices,
  # and fill it with the specified value.
  def detach3d(self, vals, mask_idxs, fill_val):
    if len(mask_idxs[0]) == 0:
      return vals
    vals = vals.clone()
    vals[mask_idxs[0], mask_idxs[1], :] = \
      vals[mask_idxs[0], mask_idxs[1], :].detach().fill_(fill_val)
    return vals

  # Detach input of (batch, ...) for the given input, for the given
  # indices, and fill it with the specified value.
  def detach2d(self, vals, mask_t, fill_val):
    if len(mask_t) == 0:
      return vals
    vals = vals.clone()
    vals[mask_t, :] = vals[mask_t, :].detach().fill_(fill_val)
    return vals

  # Softmax over unmasked idxs for each item in the batch, and pad the rest
  # with zeros. Softmax is done along dimension 0.
  # Returned tensor shape = (seq_len, batch, 1)
  def padded_softmax(self, vals, mask_idxs):
    vals = self.detach3d(vals, mask_idxs, -float('inf'))
    softmaxed = f.softmax(vals, dim=0)
    softmaxed = self.detach3d(softmaxed, mask_idxs, 0.0)
    return softmaxed

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
                             batch_size, mask_p_idxs, mask_p_ts, mask_q_idxs,
                             mask_q_ts):
    # Initial hidden and cell states for forward and backward LSTMs.
    # {h,c}{f,b}.shape = (batch, hdim / 2)
    hf, cf = self.get_initial_lstm(batch_size, self.hidden_size // 2)
    hb, cb = self.get_initial_lstm(batch_size, self.hidden_size // 2)

    # Get vectors zi for each i in passage.
    # Attended question is the same at each time step. Just compute it once.
    # attended_{question,passage}.shape = (seq_len, batch, hdim)
    attended_question = getattr(self, 'attend_question_for_passage_' + layer_no)(Hq)
    attended_passage = getattr(self, 'attend_passage_for_passage_' + layer_no)(Hpi)
    attended_question = self.detach3d(attended_question, mask_q_idxs, 0.0)
    attended_passage = self.detach3d(attended_passage, mask_p_idxs, 0.0)
    attention_q_plus_p = []
    for t in range(max_passage_len):
      attention_q_plus_p.append(attended_question + attended_passage[t])
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

      # Masking unnecessary here, as the values are already zero.
      alpha_f = f.softmax(alpha_f, dim=0)
      alpha_b = f.softmax(alpha_b, dim=0)

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
      hf = self.detach2d(hf, mask_p_ts[forward_idx], 0.0)
      hb = self.detach2d(hb, mask_p_ts[backward_idx], 0.0)
      cf = self.detach2d(cf, mask_p_ts[forward_idx], 0.0)
      cb = self.detach2d(cb, mask_p_ts[backward_idx], 0.0)

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
  def match_passage_passage(self, layer_no, Hr, max_passage_len, batch_size,
                            mask_p_idxs, mask_p_ts):
    # Initial hidden and cell states for forward and backward LSTMs.
    # {h,c}{f,b}.shape = (batch, hdim / 2)
    hf, cf = self.get_initial_lstm(batch_size, self.hidden_size // 2)
    hb, cb = self.get_initial_lstm(batch_size, self.hidden_size // 2)

    # Get vectors zi for each i in passage.
    # Attended passage is the same at each time step. Just compute it once.
    # attended_passage.shape = (seq_len, batch, hdim)
    attended_passage = getattr(self, 'attend_self_passage_' + layer_no)(Hr)
    attended_passage = self.detach3d(attended_passage, mask_p_idxs, 0.0)
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

      # Masking unnecessary here, as the values are already zero.
      alpha_f = f.softmax(alpha_f, dim=0)
      alpha_b = f.softmax(alpha_b, dim=0)

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
      hf = self.detach2d(hf, mask_p_ts[forward_idx], 0.0)
      hb = self.detach2d(hb, mask_p_ts[backward_idx], 0.0)
      cf = self.detach2d(cf, mask_p_ts[forward_idx], 0.0)
      cb = self.detach2d(cb, mask_p_ts[backward_idx], 0.0)

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
  def answer_pointer(self, Hr, Hp, Hq, mask_p_idxs, mask_p_ts, mask_q_idxs,
                     mask_q_ts, batch_size):
    # attended_input.shape = (seq_len, batch, hdim)
    attended_input = getattr(self, 'attend_input')(Hr)
    attended_input_b = getattr(self, 'attend_input_b')(Hr)
    attended_input = self.detach3d(attended_input, mask_p_idxs, 0.0)
    attended_input_b = self.detach3d(attended_input_b, mask_p_idxs, 0.0)

    # weighted_Hq.shape = (batch, hdim)
    attended_question = f.tanh(getattr(self, 'attend_question')(Hq))
    alpha_q = getattr(self, 'alpha_transform')(attended_question)
    # Padding unnecessary, as to-be masked regions are already zero.
    alpha_q = f.softmax(alpha_q, dim=0)
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
      Fk = f.tanh(attended_input + getattr(self, 'attend_answer')(ha))
      Fk_b = f.tanh(attended_input_b + getattr(self, 'attend_answer')(hb))

      # Get softmaxes over only valid paragraph lengths for each element in
      # the batch.
      # beta_k[_b].shape = (seq_len, batch, 1)
      beta_k = getattr(self, 'beta_transform')(Fk)
      beta_k_b = getattr(self, 'beta_transform')(Fk_b)

      # Mask out padded regions.
      beta_k = self.padded_softmax(beta_k, mask_p_idxs)
      beta_k_b = self.padded_softmax(beta_k_b, mask_p_idxs)

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
  def point_at_answer(self, Hr, Hp, Hq, batch_size, answer, f1_matrices,
                      mask_p_idxs, mask_p_ts, mask_q_idxs, mask_q_ts):
    # Predict the answer start and end indices.
    distribution = self.answer_pointer(Hr, Hp, Hq, mask_p_idxs, mask_p_ts,
                                       mask_q_idxs, mask_q_ts, batch_size)

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
    mle_loss = 0
    f1_loss = 0
    for idx in range(batch_size):
      loss += -torch.log(sum(batch_losses[idx]) / (1 + self.f1_loss_multiplier))
      mle_loss += -torch.log(batch_losses[idx][0] / (1 + self.f1_loss_multiplier))
      if self.f1_loss_multiplier > 0:
        f1_loss += -torch.log(batch_losses[idx][1] / (1 + self.f1_loss_multiplier))
    loss /= batch_size
    mle_loss /= batch_size
    f1_loss /= batch_size
    return distribution, loss, mle_loss, f1_loss

  # Get idxs to be padded for the given input, for the given maximum length,
  # for lengths in the batch.
  def get_mask_idxs(self, batch_size, max_len, lens):
    mask_idxs = [[], []]
    mask_ts = [ [] for _ in range(max_len) ]
    for t in range(max_len):
      for idx in range(batch_size):
        if t >= lens[idx]:
          mask_idxs[0].append(t)
          mask_idxs[1].append(idx)
          mask_ts[t].append(idx)
    return mask_idxs, mask_ts

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
    if not self.use_pretrained:
      padded_passage = self.placeholder(passage[0], False)
      padded_question = self.placeholder(question[0], False)
    batch_size = passage[0].shape[1]
    max_passage_len = passage[0].shape[0]
    max_question_len = question[0].shape[0]
    passage_lens = passage[1]
    question_lens = question[1]

    if self.debug_level >= 3:
      start_prepare = time.time()

    # Indices of values to be masked out.
    mask_p_idxs, mask_p_ts = \
      self.get_mask_idxs(batch_size, max_passage_len, passage_lens)
    mask_q_idxs, mask_q_ts = \
      self.get_mask_idxs(batch_size, max_question_len, question_lens)

    # Get embedded passage and question representations.
    if not self.use_pretrained:
      p = torch.transpose(self.embedding(torch.t(padded_passage)), 0, 1)
      q = torch.transpose(self.embedding(torch.t(padded_question)), 0, 1)
    else:
      p = self.get_vector_embeddings(passage[0])
      q = self.get_vector_embeddings(question[0])

    # {p,q}.shape = (seq_len, batch, embedding_dim + num_pos_tags + num_ner_tags)
    p = torch.cat((p, self.placeholder(passage_pos_tags),
                   self.placeholder(passage_ner_tags)), dim=-1)
    q = torch.cat((q, self.placeholder(question_pos_tags),
                   self.placeholder(question_ner_tags)), dim=-1)

    if self.debug_level >= 3:
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

    if self.debug_level >= 3:
      Hp.sum()
      Hq.sum()
      print "Data pre-processing time: %.2fs" % (time.time() - start_preprocess)
      start_matching = time.time()

    # Bi-directional multi-layer MatchLSTM for question-aware passage representation.
    Hr = Hp
    for layer_no in range(self.num_matchlstm_layers):
      Hr = self.match_question_passage(str(layer_no), Hr, Hq, max_passage_len,
                                       batch_size, mask_p_idxs, mask_p_ts,
                                       mask_q_idxs, mask_q_ts)
      # Question-aware passage representation dropout.
      Hr = getattr(self, 'dropout_passage_matchlstm_' + str(layer_no))(Hr)

    if self.debug_level >= 3:
      Hr.sum()
      print "Matching passage with question time: %.2fs" % \
            (time.time() - start_matching)
      start_postprocess = time.time()

    # (Question-aware) passage self-matching layers.
    for layer_no in range(self.num_selfmatch_layers):
      Hr = self.match_passage_passage(str(layer_no), Hr, max_passage_len,
                                      batch_size, mask_p_idxs, mask_p_ts)
      # Passage self-matching layer dropout.
      Hr = getattr(self, 'dropout_self_matchlstm_' + str(layer_no))(Hr)

    if self.debug_level >= 3 and self.num_selfmatch_layers > 0:
      Hr.sum()
      print "Self-matching time: %.2fs" % (time.time() - start_postprocess)
      start_postprocess = time.time()

    if self.num_postprocessing_layers > 0:
      Hr = self.process_input_with_lstm(Hr, max_passage_len, passage_lens, batch_size,
                                        self.postprocessing_lstm)

    if self.debug_level >= 3:
      Hr.sum()
      print "Post-processing question-aware passage time: %.2fs" % \
            (time.time() - start_postprocess)
      start_answer = time.time()

    # Get probability distributions over the answer start, answer end,
    # and the loss for training.
    # At this point, Hr.shape = (seq_len, batch, hdim)
    answer_distributions_list, loss, mle_loss, f1_loss = \
      self.point_at_answer(Hr, Hp, Hq, batch_size, answer, f1_matrices,
                           mask_p_idxs, mask_p_ts, mask_q_idxs, mask_q_ts)

    if self.debug_level >= 3:
      loss.data[0]
      print "Answer pointer time: %.2fs" % (time.time() - start_answer)

    self.loss = loss
    self.mle_loss = mle_loss
    self.f1_loss = f1_loss
    return answer_distributions_list

