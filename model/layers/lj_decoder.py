import random
import torch
from torch import nn
from torch.nn import functional as F
from layers.rnncells import StackedLSTMCell, StackedGRUCell
from layers.beam_search import Beam
#from .feedforward import FeedForward
from utils import to_var, SOS_ID, UNK_ID, EOS_ID,PAD_ID
from scipy.stats import  pearsonr
import math
from typing import Union, List
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-31

class BaseRNNDecoder(nn.Module):
    def __init__(self):
        """Base Decoder Class"""
        super(BaseRNNDecoder, self).__init__()

    @property
    def use_lstm(self):
        return isinstance(self.rnncell, StackedLSTMCell)

    def init_token(self, batch_size, SOS_ID=SOS_ID):
        """Get Variable of <SOS> Index (batch_size)"""
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            # (h, c)
            return (to_var(torch.zeros(self.num_layers,
                                       batch_size,
                                       self.hidden_size)),
                    to_var(torch.zeros(self.num_layers,
                                       batch_size,
                                       self.hidden_size)))
        else:
            # h
            return to_var(torch.zeros(self.num_layers,
                                      batch_size,
                                      self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
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

    def decode(self, out):
        """
        Args:
            out: unnormalized word distribution [batch_size, vocab_size]
        Return:
            x: word_index [batch_size]
        """

        # Sample next word from multinomial word distribution
        if self.sample:
            # x: [batch_size] - word index (next input)
            x = torch.multinomial(self.softmax(out / self.temperature), 1).view(-1)

        # Greedy sampling
        else:
            # x: [batch_size] - word index (next input)
            _, x = out.max(dim=1)
        return x

    def linear(self):
        raise NotImplementedError

    def forward(self):
        """Base forward function to inherit"""
        raise NotImplementedError

    def forward_step(self):
        """Run RNN single step"""
        raise NotImplementedError

    def embed(self, x):
        """word index: [batch_size] => word vectors: [batch_size, hidden_size]"""

        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0))))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed

    def beam_decode(self, emotion,behavior,encoder_outputs,input_ids_tensor,ext_vocab_size,init_h=None,sample=True,pointer=True):
        """
        Args:
            encoder_outputs (Variable, FloatTensor): [batch_size, source_length, hidden_size]
            input_valid_length (Variable, LongTensor): [batch_size] (optional)
            init_h (variable, FloatTensor): [batch_size, hidden_size] (optional)
        Return:
            out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(h=init_h)

        # [batch_size x beam_size]
        x = self.init_token(batch_size * self.beam_size, SOS_ID)

        # [num_layers, batch_size x beam_size, hidden_size]
        decoder_hidden = self.init_h(batch_size, hidden=init_h).repeat(1, self.beam_size, 1)
        emotion = self.init_h(batch_size, hidden=emotion).repeat(1, self.beam_size, 1)
        encoder_outputs = self.init_h(batch_size, hidden=encoder_outputs).repeat( self.beam_size,1, 1)
        input_ids_tensor = self.init_h(batch_size, hidden=input_ids_tensor).repeat(self.beam_size,  1)
        # batch_position [batch_size]
        #   [0, beam_size, beam_size * 2, .., beam_size * (batch_size-1)]
        #   Points where batch starts in [batch_size x beam_size] tensors
        #   Ex. position_idx[5]: when 5-th batch starts
        batch_position = to_var(torch.arange(0, batch_size).long() * self.beam_size)

        # Initialize scores of sequence
        # [batch_size x beam_size]
        # Ex. batch_size: 5, beam_size: 3
        # [0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf, 0, -inf, -inf]
        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.beam_size, 0.0)
        score = to_var(score)

        # Initialize Beam that stores decisions for backtracking
        beam = Beam(
            batch_size,
            self.hidden_size,
            ext_vocab_size,#self.vocab_size,
            self.beam_size,
            self.max_unroll,
            batch_position)
        decoder_states=[]
        log_prob = not (sample or pointer)
        enc_attn_weights = []
        # coverage_losses: Union[torch.Tensor, float] = 0
        coverage_losses = torch.zeros([1, ]).cuda()
        for i in range(self.max_unroll):

            # x: [batch_size x beam_size]; (token index)
            # =>
            # out: [batch_size x beam_size, vocab_size]
            # h: [num_layers, batch_size x beam_size, hidden_size]
            # decoder_hidden = torch.cat([decoder_hidden, emotion], dim=2)
            if enc_attn_weights:
                coverage_vector = self.get_coverage_vector(enc_attn_weights)
            else:
                coverage_vector = None
            decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = self.forward_step(x, decoder_hidden, encoder_outputs,
                                                                            torch.cat(decoder_states) if decoder_states else None,
                                                                                emotion,coverage_vector,encoder_word_idx=input_ids_tensor,
                                                                                ext_vocab_size=ext_vocab_size,log_prob=log_prob)# log_prob: [batch_size x beam_size, vocab_size]
            if self.enc_attn_cover or self.cover_loss > 0:
                if coverage_vector is not None and self.cover_loss > 0:
                    coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size \
                                    * self.cover_loss
                    coverage_losses = coverage_losses + coverage_loss
                    # if include_cover_loss: r.loss_value += coverage_loss.item()
                enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
            if self.dec_attn:
                decoder_states.append(decoder_hidden)
            log_prob = F.log_softmax(decoder_output, dim=1)/self.max_unroll

            # [batch_size x beam_size]
            # => [batch_size x beam_size, vocab_size]
            score = score.view(-1, 1) + log_prob

            # Select `beam size` transitions out of `vocab size` combinations

            # [batch_size x beam_size, vocab_size]
            # => [batch_size, beam_size x vocab_size]
            # Cutoff and retain candidates with top-k scores
            # score: [batch_size, beam_size]
            # top_k_idx: [batch_size, beam_size]
            #       each element of top_k_idx [0 ~ beam x vocab)

            score, top_k_idx = score.view(batch_size, -1).topk(self.beam_size, dim=1)
            # topk_k_idx是在前面三个为输入的情况下后面的概率的最大值，可能最大的几个词出现在一个beam里面，因为现在是beamsize的句子拼接在了一起，所以才有beamidx
            # Get token ids with remainder after dividing by top_k_idx
            # Each element is among [0, vocab_size)
            # Ex. Index of token 3 in beam 4
            # (4 * vocab size) + 3 => 3
            # x: [batch_size x beam_size]
            x = (top_k_idx % ext_vocab_size).view(-1)#top_k_idx为什么会大于vocab_size

            # top-k-pointer [batch_size x beam_size]
            #       Points top-k beam that scored best at current step
            #       Later used as back-pointer at backtracking
            #       Each element is beam index: 0 ~ beam_size
            #                     + position index: 0 ~ beam_size x (batch_size-1)
            beam_idx = top_k_idx // ext_vocab_size # [batch_size, beam_size] 每一个元素的beamindex？
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)

            # Select next h (size doesn't change)
            # [num_layers, batch_size * beam_size, hidden_size]
            decoder_hidden = decoder_hidden.index_select(1, top_k_pointer)

            # Update sequence scores at beam
            beam.update(score.clone(), top_k_pointer, x)  # , h)

            # Erase scores for EOS so that they are not expanded
            # [batch_size, beam_size]
            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))

        # prediction ([batch, k, max_unroll])
        #     A list of Tensors containing predicted sequence
        # final_score [batch, k]
        #     A list containing the final scores for all top-k sequences
        # length [batch, k]
        #     A list specifying the length of each sequence in the top-k candidates
        # prediction, final_score, length = beam.backtrack()
        prediction, final_score, length = beam.backtrack()

        return coverage_losses/self.max_unroll,prediction, final_score, length


class DecoderRNN(BaseRNNDecoder):
    def __init__(self, vocab_size, vocab,embedding_size,
                 hidden_size,D_e, rnncell=StackedGRUCell, num_layers=1,
                 enc_attn=True, dec_attn=False,enc_attn_cover=True, pointer=True, cover_func=None,cover_loss=None,
                 dropout=0.0, word_drop=0.0,out_drop=0.0,
                 max_unroll=40, sample=True, temperature=1.0, beam_size=1, enc_hidden_size=None):
        super(DecoderRNN, self).__init__()
        self.D_e=D_e
        self.vocab=vocab
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.word_drop=word_drop
        self.temperature = temperature
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size
        #self.hidden_size = hidden_size
        self.combined_size = self.hidden_size
        self.enc_attn = enc_attn
        self.dec_attn = dec_attn
        self.enc_attn_cover = enc_attn_cover
        self.pointer = pointer
        self.cover_loss=cover_loss
        self.cover_func=cover_func
        # self.w_a = nn.Bilinear(300,300, 1)  # 双线性函数(y = x1 * A * x2 + b)

        # self.w1=torch.nn.init.normal_(torch.zeros((300,32),requires_grad=True).cuda(), mean=0, std=1)
        #self.out_embed_size = out_embed_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.softmax = nn.Softmax(dim=1)
        self.linear_ = nn.Linear(self.hidden_size, self.hidden_size)
        if enc_attn:
            if not enc_hidden_size: enc_hidden_size = self.hidden_size
            self.enc_bilinear = nn.Bilinear(self.hidden_size, enc_hidden_size, 1)  # 双线性函数(y = x1 * A * x2 + b)
            self.emo_bilinear = nn.Bilinear(self.hidden_size, self.D_e, 1)  # 双线性函数(y = x1 * A * x2 + b)
            print("size:",self.hidden_size, enc_hidden_size)
            self.combined_size += enc_hidden_size
            # self.combined_size += self.hidden_size#emotion
            if enc_attn_cover:
                self.cover_weight = nn.Parameter(torch.rand(1))

        if dec_attn:
            self.dec_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
            self.combined_size += self.hidden_size
        self.rnncell = rnncell(num_layers,
                               self.combined_size,
                               hidden_size,
                               dropout)
        self.out_drop = nn.Dropout(out_drop) if out_drop > 0 else None
        if pointer:
            self.ptr = nn.Linear(self.combined_size+self.D_e, 1)
        if embedding_size != self.combined_size:
            # use pre_out layer if combined size is different from embedding size
            self.out_embed_size = embedding_size
        if self.out_embed_size:  # use pre_out layer
            self.pre_out = nn.Linear(self.hidden_size+self.D_e, self.out_embed_size)
            size_before_output = self.out_embed_size
        else:  # don't use pre_out layer
            size_before_output = self.combined_size

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
        self.out = nn.Linear(size_before_output, vocab_size)
        self.emo_gate=nn.Linear(self.combined_size+self.hidden_size,self.D_e)
    def filter_oov(self, tensor, ext_vocab_size):
        """Replace any OOV index in `tensor` with UNK"""
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = UNK_ID
            return result
        return tensor

    def get_coverage_vector(self, enc_attn_weights):
        """Combine the past attention weights into one vector"""
        if self.cover_func == 'max':
            coverage_vector, _ = torch.max(torch.cat(enc_attn_weights), dim=0)
        elif self.cover_func == 'sum':
            coverage_vector = torch.sum(torch.cat(enc_attn_weights), dim=0)
        else:
            raise ValueError('Unrecognized cover_func: ' + self.cover_func)
        return coverage_vector

    def forward_step(self, x, last_hidden, encoder_outputs=None, decoder_states=None, emotion=None,coverage_vector=None, *,
                     encoder_word_idx=None, ext_vocab_size: int = None, log_prob: bool = True):
        """
        Single RNN Step
        1. Input Embedding (vocab_size => hidden_size)
        2. RNN Step (hidden_size => hidden_size)
        3. Output Projection (hidden_size => vocab size)

        Args:
            x: [batch_size]
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)

        Return:
            out: [batch_size,vocab_size] (Unnormalized word distribution)
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        # x: [batch_size] => [batch_size, hidden_size]
        # hidden = self.linear_(hidden)
        # hidden=F.tanh(hidden)
        batch_size = x.size(0)
        combined = torch.zeros(batch_size, self.combined_size, device=DEVICE)  # 用于存放

        offset = self.embedding_size
        enc_attn, prob_ptr = None, None
        # Unormalized word distribution
        # out: [batch_size, vocab_size]
        encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
        # emo-attention
        embedding = self.embedding(self.filter_oov(x, ext_vocab_size))
        # last_h: [batch_size, hidden_size] (h from Top RNN layer)
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        # last_h, hidden = self.rnncell(embedding, hidden)

        # if self.use_lstm:
            # last_h_c: [2, batch_size, hidden_size] (h from Top RNN layer)
            # h_c: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
            # last_h = last_h[0]
        if self.enc_attn or self.pointer:
            num_enc_steps = encoder_outputs.size(0)
            enc_total_size = encoder_outputs.size(2) + emotion.size(2)
            enc_energy = self.enc_bilinear(last_hidden.expand(num_enc_steps, batch_size, -1).contiguous(),
                                           encoder_outputs)  # 计算decoderhidden和每一个encoderoutputs的关系（权重）

            enc_energy = F.tanh(enc_energy)
            if self.enc_attn_cover and coverage_vector is not None:
                enc_energy = enc_energy + self.cover_weight * torch.log(
                    coverage_vector.transpose(0, 1).unsqueeze(2) + eps)
            # transpose => (batch size, num encoder states, 1)
            enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1)  # 求出注意力权重
            # a=emo_energy.expand(1, batch_size, emotion.size(2))
            if self.enc_attn:
                # context: (batch size, encoder hidden size, 1)
                enc_context = torch.bmm(encoder_outputs.permute(1, 2, 0), enc_attn)  # 对encoder_outputs加权求和
                # com_context=torch.cat([enc_context,emo_context.permute(1, 2, 0)],dim=1)
                combined[:, offset:offset + enc_total_size] = enc_context.squeeze(2)
                offset += enc_total_size
            enc_attn = enc_attn.squeeze(2)
        if self.dec_attn:
            if decoder_states is not None and len(decoder_states) > 0:
                dec_energy = self.dec_bilinear(last_hidden.expand_as(decoder_states).contiguous(),
                                               decoder_states)
                dec_attn = F.softmax(dec_energy, dim=0).transpose(0, 1)
                dec_context = torch.bmm(decoder_states.permute(1, 2, 0), dec_attn)
                combined[:, offset:offset + self.hidden_size] = dec_context.squeeze(2)
            offset += self.hidden_size
        # decoder RNN
        combined[:, :self.embedding_size] = embedding.squeeze(0)  # as RNN expects a 3D tensor (step=1)
        last_h, hidden = self.rnncell(combined, last_hidden)
        emo_energy = self.emo_bilinear(last_hidden, emotion)
        emo_energy = F.tanh(emo_energy)
        emo_context = torch.mul(emo_energy.expand(1, batch_size, emotion.size(2)), emotion)  # 对应为相乘
        out=torch.cat([hidden,emo_context],dim=-1)
        # hidden_emo=hidden+emo_context.squeeze(0)
        # emo_gate=F.sigmoid(self.emo_gate(torch.cat([combined,last_hidden.squeeze(0)],dim=1)))
        # emotion=torch.mul(emo_gate,emotion.squeeze(0))
        # combined_ = torch.cat([combined, emo_context.squeeze(0)], dim=1)
        # last_h: [batch_size, hidden_size] (h from Top RNN layer)
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)

        #
        if self.use_lstm:
        # last_h_c: [2, batch_size, hidden_size] (h from Top RNN layer)
        # h_c: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
            last_h = last_h[0]
        if self.out_embed_size:
            out_embed = self.pre_out(out)
        else:
            out_embed = combined
        logits = self.out(out_embed)
        if self.pointer:
            output = torch.zeros(batch_size, ext_vocab_size, device=DEVICE)
            # distribute probabilities between generator and pointer
            prob_ptr = F.sigmoid(self.ptr(torch.cat([combined,last_hidden.squeeze(0)],dim=1)))  # (batch size, 1)在句子中选词的概率
            prob_gen = 1 - prob_ptr  # 从词表中选词
            # add generator probabilities to output
            #    gen_output = F.softmax(logits, dim=1)  # can't use log_softmax due to adding probabilities
            output[:, :self.vocab_size] = prob_gen * logits  # gen_output
            # add pointer probabilities to output
            ptr_output = enc_attn
            output.scatter_add_(1, encoder_word_idx,
                                prob_ptr * ptr_output)  # self_tensor.scatter_add_(dim, index_tensor, other_tensor) → 输出tensor——将other_tensor中的数据，按照index_tensor中的索引位置，添加至self_tensor矩阵中。
        #    if log_prob: output = torch.log(output + eps)
        else:
            output=logits
        return output, hidden, enc_attn, prob_ptr

    def forward(self, inputs,emotion,encoder_outputs,input_ids_tensor,ext_vocab_size,init_h=None, input_valid_length=None,
                decode=False):
        """
        Train (decode=False)
            Args:
                inputs (Variable, LongTensor): [batch_size, seq_len]
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len, vocab_size]
        Test (decode=True)
            Args:
                inputs: None
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(inputs, init_h)
        content=init_h
        # x: [batch_size]
        x = self.init_token(batch_size, SOS_ID)

        # h: [num_layers, batch_size, hidden_size]
        decoder_hidden = self.init_h(batch_size, hidden=content)#RNN初始化状态,hidden有值直接返回的是hidden中的值
        #e=self.init_h(batch_size,hidden=emotion)
        log_prob = not (self.sample or self.pointer)
        if not decode:
            out_list = []
            seq_len = inputs.size(1)#30
            decoder_states = []
            enc_attn_weights = []
            # coverage_losses:Union[torch.Tensor, float]=0
            coverage_losses=torch.zeros([1,]).cuda()#to_var(torch.FloatTensor(coverage_losses))
            for i in range(seq_len):#每个人词的生成操作
                if enc_attn_weights:
                    coverage_vector = self.get_coverage_vector(enc_attn_weights)
                else:
                    coverage_vector = None
                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = self.forward_step(x, decoder_hidden, encoder_outputs,
                     torch.cat(decoder_states) if decoder_states else None,emotion, coverage_vector=coverage_vector,
                     encoder_word_idx=input_ids_tensor, ext_vocab_size=ext_vocab_size,
                     log_prob=log_prob)
                if self.enc_attn_cover or self.cover_loss > 0:
                    if coverage_vector is not None and self.cover_loss > 0:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size \
                                        * self.cover_loss
                        coverage_losses = coverage_losses + coverage_loss
                        # if include_cover_loss: r.loss_value += coverage_loss.item()
                    enc_attn_weights.append(dec_enc_attn.unsqueeze(0))

                if self.dec_attn:
                    decoder_states.append(decoder_hidden)
                out_list.append(decoder_output)
                x = inputs[:, i]

            # [batch_size, max_target_len, vocab_size]
            return coverage_losses/self.max_unroll,torch.stack(out_list, dim=1)#,coverage_losses
        else:
            x_list = []
            decoder_states = []
            enc_attn_weights=[]
            coverage_losses=torch.zeros([1,]).cuda()
            # coverage_vector=None
            for i in range(self.max_unroll):
                if enc_attn_weights:
                    coverage_vector = self.get_coverage_vector(enc_attn_weights)
                else:
                    coverage_vector = None
                decoder_output, decoder_hidden, dec_enc_attn, dec_prob_ptr = self.forward_step(x, decoder_hidden,
                                                                                               encoder_outputs,
                                                                                               torch.cat(decoder_states) if decoder_states else None,
                                                                                               emotion,
                                                                                               coverage_vector=coverage_vector,
                                                                                               encoder_word_idx=input_ids_tensor,
                                                                                               ext_vocab_size=ext_vocab_size,
                                                                                               log_prob=log_prob)
                if self.dec_attn:
                    decoder_states.append(decoder_hidden)
                if self.enc_attn_cover or self.cover_loss > 0:
                    if coverage_vector is not None and self.cover_loss > 0:
                        coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size \
                                        * self.cover_loss
                        coverage_losses = coverage_losses + coverage_loss
                        # if include_cover_loss: r.loss_value += coverage_loss.item()
                    enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
                # if self.enc_attn_cover or self.cover_loss > 0:
                #     if coverage_vector is not None and self.cover_loss > 0:
                #         coverage_loss = torch.sum(torch.min(coverage_vector, dec_enc_attn)) / batch_size \
                #                         * self.cover_loss
                #         coverage_losses = coverage_losses + coverage_loss
                        # if include_cover_loss: r.loss_value += coverage_loss.item()
                    # enc_attn_weights.append(dec_enc_attn.unsqueeze(0))
                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                output=F.softmax(decoder_output)
                x = self.decode(output)
                x_list.append(x)

            # [batch_size, max_target_len]
            return coverage_losses/self.max_unroll,torch.stack(x_list, dim=1)#one sentence coverageloss

