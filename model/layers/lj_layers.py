import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .lj_attention import SelfAttention,SimpleAttention,MatchingAttention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from utils import to_var, reverse_order_valid, PAD_ID,UNK_ID
from utils import Vocab
if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor



class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size)),
                    to_var(torch.zeros(self.num_layers*self.num_directions,
                                      batch_size,
                                      self.hidden_size)))
        else:
            return to_var(torch.zeros(self.num_layers*self.num_directions,
                                        batch_size,
                                        self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError

class EncoderRNN(BaseRNNEncoder):
    def __init__(self, self_attention_hidden_size,self_attention_heads,
                 self_attention_dropout,vocab_size,vocab, embedding_size,
                 hidden_size, rnn=nn.GRU, num_layers=1, bidirectional=False,
                 dropout=0.0, bias=True, batch_first=True):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        #self.vocab=vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.self_attention_hidden_size=self_attention_hidden_size
        self.self_attention_heads=self_attention_heads
        self.self_attention_dropout=self_attention_dropout
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.vocab=vocab
        # word embedding
        if vocab.embeddings is not None:
            self.embedding_size = vocab.embeddings.shape[1]
            if embedding_size is not None and self.embedding_size != embedding_size:
                print("Warning: Model embedding size %d is overriden by pre-trained embedding size %d."
                      % (embedding_size, self.embed_size))
            embedding_weights = torch.from_numpy(vocab.embeddings)
        else:
            self.embed_size = embedding_size
            embedding_weights = None
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID,_weight=embedding_weights)
        self.rnn = rnn(input_size=embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)
        # self.w_s=nn.Linear(hidden_size,1)
        # self.convert=nn.Linear(hidden_size*2,hidden_size)
        # self.content_self_attention=SelfAttention(hidden_size=self_attention_hidden_size,
        #                                  num_attention_heads=self_attention_heads,
        #                                  dropout_prob=self_attention_dropout)
        # self.emotion_self_attention=SelfAttention(hidden_size=self_attention_hidden_size,
        #                                  num_attention_heads=self_attention_heads,
        #                                  dropout_prob=self_attention_dropout)

    def filter_oov(self, tensor, ext_vocab_size):
        """Replace any OOV index in `tensor` with UNK"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = UNK_ID
            return result
        return tensor

    def forward(self,ext_vocab_size, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [conv_length,batchsize,2, seq_len]
            input_length (Variable, LongTensor):[conv_length,batchsize,2]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()
        mask=[]
        for sen in inputs:
            seq_mask=[]
            for m in sen:
               if m != 0:
                   seq_mask.append(1)
               else:
                   seq_mask.append(0)
            mask.append(seq_mask)
        mask = to_var(torch.FloatTensor(mask))

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.data.tolist()

        # [num_sentences, max_source_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_sentences, max_source_length, embedding_dim]
        embedded = self.embedding(self.filter_oov(inputs_sorted, ext_vocab_size))

        # batch_first=True rnn_input为PackedSequence2类型 包括了batchsize（tensor40）以及data（tensor）
        rnn_input = pack_padded_sequence(embedded, input_length_sorted,
                                         batch_first=self.batch_first)
        #print(rnn_input.batch_sizes)
        hidden = self.init_h(batch_size, hidden=hidden)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        self.rnn.flatten_parameters()
        #我的理解是，为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。类似我们调用tensor.contiguous
        outputs, all_hidden = self.rnn(rnn_input, hidden)#hidden是最后的输出，output是没个时刻的输出
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first,total_length=seq_len)
        #print(outputs.size())
        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)
        hidden=torch.cat((all_hidden[-2].unsqueeze(0),all_hidden[-1].unsqueeze(0)),dim=0)
        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                      hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)
        return hidden,outputs

class ContextRNN(BaseRNNEncoder):
    def __init__(self, input_size, context_size, rnn=nn.GRU, num_layers=1, dropout=0.0,
                 bidirectional=False, bias=True, batch_first=True):
        """Context-level Encoder"""
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(input_size=input_size,
                        hidden_size=context_size,
                        num_layers=num_layers,
                        bias=bias,
                        batch_first=batch_first,
                        dropout=dropout,
                        bidirectional=bidirectional)

    def forward(self, encoder_hidden, conversation_length, hidden=None):#传过来的数据
        """
        Args:
            encoder_hidden (Variable, FloatTensor): [batch_size, max_len, num_layers * direction * hidden_size]
            conversation_length (Variable, LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_seq_len, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len, _  = encoder_hidden.size()

        # Sort for PackedSequenceindices为 原来对话所在位置？
        conv_length_sorted, indices = conversation_length.sort(descending=True)
        conv_length_sorted = conv_length_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)#将hidden按照sorted的次序排列

        rnn_input = pack_padded_sequence(encoder_hidden_sorted, conv_length_sorted, batch_first=True)

        hidden = self.init_h(batch_size, hidden=hidden)#rnn初始化

        self.rnn.flatten_parameters()#gpu上才执行，重置参数数据指针，以便使用更快的代码路径。
        outputs, hidden = self.rnn(rnn_input, hidden)

        # outputs: [batch_size, max_conversation_length, context_size]
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (hidden[0].index_select(1, inverse_indices),
                    hidden[1].index_select(1, inverse_indices))
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):

        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden

class FeatureExtractionCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,behavior_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(FeatureExtractionCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m,D_g)#全局GRU
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)#用户状态GRU
        self.e_cell = nn.GRUCell(D_p,D_e)#情感GRU
        self.c_cell = nn.GRUCell(D_g, D_g)#内容GRU
        # if behavior_state==True:
        #     self.b_cell = nn.GRUCell(D_p,D_b)

        self.dropout = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
            # self.attention_2 = SimpleAttention(D_g)
            self.content_attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)
            # self.attention_2 = MatchingAttention(D_g, D_m, D_a, context_attention)
            self.content_attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0,con0,a0,re0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        R_u=U[:,1,:]
        P_u=U[:,0,:]
        R_qm=qmask[:,1,:]
        P_qm=qmask[:,0,:]
        q0 = torch.zeros(P_qm.size()[0], self.D_p).type(P_u.type()) if q0.size()[0] == 0 \
            else q0
        a0 = torch.zeros(R_qm.size()[0], self.D_p).type(R_u.type()) if a0.size()[0] == 0 \
            else a0
        P_u_g_ = self.g_cell(P_u, #torch.cat([P_u,q0],dim=-1)
                torch.zeros((U.size()[0],self.D_g)).type(P_u.type()) if g_hist.size()[0]==0 else
                g_hist[-1])#取最近的全局状态，当前句子和前一时刻全局状态送入Global GRU
        P_u_g_ = self.dropout(P_u_g_)
        R_u_g_ = self.g_cell(R_u, P_u_g_)
        R_u_g_ = self.dropout(R_u_g_)
        g_ = torch.cat([P_u_g_.unsqueeze(0), R_u_g_.unsqueeze(0)], dim=0)
        if g_hist.size()[0] == 0:  # 第一句话
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist, P_u)
        g_hist=torch.cat([g_hist,P_u_g_.unsqueeze(0)],dim=0)
        rc_, ralpha = self.attention(g_hist, R_u)
        U_c_ = torch.cat([P_u,c_], dim=1)
        rU_c_=torch.cat([R_u,rc_],dim=1)

        qs_ = self.p_cell(U_c_.contiguous(),a0).view(P_u.size()[0],self.D_p)#矫正维度
        q_ = self.dropout(qs_)
        as_ = self.p_cell(rU_c_.contiguous(), q_).view(R_u.size()[0], self.D_p) # 矫正维度
        a_ = self.dropout(as_)
        re0 = torch.zeros(R_qm.size()[0], self.D_e).type(R_u.type()) if re0.size()[0] == 0 \
            else re0
        e0 = torch.zeros(P_qm.size()[0], self.D_e).type(P_u.type()) if e0.size()[0]==0\
                else e0#前一时刻的e
        e_ = self.e_cell(q_,e0)
        e_ = self.dropout(e_)
        re_ = self.e_cell(a_, re0)
        re_ = self.dropout(re_)
        #内容
        con0 = torch.zeros(P_qm.size()[0], self.D_g).type(P_u.type()) if con0.size()[0] == 0 \
            else con0  # 前一时刻的e
        con_ = self.c_cell(P_u_g_, con0)
        con_ = self.dropout(con_)  # 只有用户的情感
        return g_,q_,e_,c_,con_,alpha,a_,re_




class FeatureEtraction(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,behavior_state=False,
                            context_attention='simple', D_a=100, dropout=0.5):
        super(FeatureEtraction, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)
        self.feature_cell = FeatureExtractionCell(D_m, D_g, D_p, D_e,
                            listener_state,behavior_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor global_history
        c_hist = torch.zeros(0).type(U.type())
        q_ = torch.zeros(0).type(U.type())#query
        a_ = torch.zeros(0).type(U.type())#answer
        e_ = torch.zeros(0).type(U.type())
        e = e_
        con_ = torch.zeros(0).type(U.type())
        con = con_
        a=a_
        q=q_
        re_ = torch.zeros(0).type(U.type())
        re = e_

        alpha = []#attention权重
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, c_,con_,alpha_,a_,re_, = self.feature_cell (u_, qmask_, g_hist, q_, e_,con_,a_,re_)
            g_hist = torch.cat([g_hist, g_],0)
            c_hist = torch.cat([c_hist, c_.unsqueeze(0)], 0)#q用户情感状态,c是attention后的特征
            # c_a_hist = torch.cat([c_a_hist, c_a.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            a = torch.cat([a,a_.unsqueeze(0)],0)
            re = torch.cat([re, re_.unsqueeze(0)], 0)
            # if b_ != None:
            #     b = torch.cat([b, b_.unsqueeze(0)], 0)
            #     rb = torch.cat([b, rb_.unsqueeze(0)], 0)
            con = torch.cat([con, con_.unsqueeze(0)], 0)
            q = torch.cat([q,q_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return con,e,re,q,a# seq_len, batch, D_e#内容只通过全局状态dedao，c_hist是通过注意力机制分配权重后的全局变量加权和，两者可做对比。