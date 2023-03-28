import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from utils import to_var, pad, normal_kl_div, normal_logpdf, bag_of_words_loss, to_bow, EOS_ID
import layers.lj_layers as lj_layers
import layers.lj_decoder as lj_decoder
from utils.vocab import Vocab
from torch.autograd import Variable
import numpy as np

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class ETHREED(nn.Module):
    def __init__(self, config):
        super(ETHREED, self).__init__()

        self.config = config
        print('Loading Vocabulary...')
        vocab = Vocab()
        vocab.load(config.word2id_path, config.id2word_path)
        if config.glove_path is not None:
            # vocab=torch.load(config.glove_path)
            vocab.load_embeddings(config.glove_path)
        print(f'Vocabulary size: {vocab.vocab_size}')

        config.vocab_size = vocab.vocab_size
        # config.ext_vocab_size=vocab.vocab_size+vocab.oov_size
        self.dropout=nn.Dropout(self.config.dropout)
        self.encoder = lj_layers.EncoderRNN(config.self_attention_hidden_size,
                                         config.self_attention_head,
                                         config.self_attention_dropout,
                                         config.vocab_size,
                                         vocab,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout
                                        )

        context_input_size = (1#config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)
        self.context_encoder = lj_layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 1,#config.num_layers,
                                                 config.dropout)
        self.feature_extraction=lj_layers.FeatureEtraction(config.D_m,
                                                           config.D_g,
                                                           config.D_p,
                                                           config.D_e,
                                                           config.listener_state,
                                                           config.behavior_state,
                                                           config.context_attention,
                                                           config.D_a,
                                                           config.dropout_rec)
        if config.listener_state==True :
            if config.behavior_state==True:
                self.l_cell=nn.GRUCell(config.D_e+config.D_b,config.D_r)
            else:
                self.l_cell = nn.GRUCell(config.D_e, config.D_r)
        self.decoder = lj_decoder.DecoderRNN(config.vocab_size,
                                             vocab,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                             config.D_e,
                                         config.rnncell,
                                         1,#config.num_layers,
                                         config.enc_attn,
                                         config.dec_attn,
                                         config.enc_attn_cover,
                                         config.pointer,
                                         config.cover_func,
                                         config.cover_loss,
                                         config.dropout,
                                         config.word_drop,
                                         config.out_drop,
                                         config.max_unroll,
                                         config.sample,
                                         config.temperature,
                                         config.beam_size,
                                         config.enc_total_size)

        self.context2decoder = layers.FeedForward(config.context_size,
                                                  1* config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)
        self.context2decoder_emotion = layers.FeedForward(config.D_e,config.D_e,
                                                  num_layers=1,
                                                  activation=config.activation)

        self.emotion2pre = layers.FeedForward(config.D_e,
                                                  1 * config.D_e,
                                                  num_layers=1,
                                                  activation=config.activation)
        self.emo_classification = nn.Linear(self.config.D_e, self.config.n_classes).cuda()
        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding
        self.softplus = nn.Softplus()
        if self.config.emotion_policy== True:
            self.prior_h = layers.FeedForward(config.D_e,
                                          config.D_e,
                                          num_layers=1,
                                          hidden_size=config.D_e,
                                          activation=config.activation)
            self.prior_mu = nn.Linear(config.D_e,
                                  config.D_e)
            self.prior_var = nn.Linear(config.D_e,
                                   config.D_e)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

        if config.bow:
            self.bow_h = layers.FeedForward(config.z_sent_size,
                                            config.decoder_hidden_size,
                                            num_layers=1,
                                            hidden_size=config.decoder_hidden_size,
                                            activation=config.activation)
            self.bow_predict = nn.Linear(config.decoder_hidden_size, config.vocab_size)

    def prior(self, context_outputs):
        # Context dependent prior
        h_prior = self.prior_h(context_outputs)
        mu_prior = self.prior_mu(h_prior)
        var_prior = self.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior

    def posterior(self, context_outputs, encoder_hidden):
        h_posterior = self.posterior_h(torch.cat([context_outputs, encoder_hidden], 1))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def compute_bow_loss(self, target_conversations):
        target_bow = np.stack([to_bow(sent, self.config.vocab_size) for conv in target_conversations for sent in conv], axis=0)
        target_bow = to_var(torch.FloatTensor(target_bow))
        bow_logits = self.bow_predict(self.bow_h(self.z_sent))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss
    def emp_policy(self,P_sen_emotions,input_conversation_length,decode=False):
        P_emotions_input = torch.cat([P_sen_emotions[i, :int(l / 2), :]
                                      for i, l in enumerate(input_conversation_length.data)])
        P_emotions_input = self.emotion2pre(P_emotions_input)
        num_sentiments = P_emotions_input.size(0)
        mu_prior, var_prior = self.prior(P_emotions_input)
        eps = to_var(torch.randn((int(num_sentiments), self.config.D_e)))
        R_emotions_prior = mu_prior + torch.sqrt(var_prior) * eps
        kl_div=None
        return R_emotions_prior,kl_div
    # def behavior_value_network(self,R_behavior,emo,states):
    def forward(self, batch_size,ext_vocab_size,input_sentences, input_sentence_length,
                input_conversation_turns,input_conversation_length, target_sentences,speakers, decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """

        num_sentences = input_sentences.size(0)#numpy类的函数
        sen_len=input_sentences.size(1)
        max_len = input_conversation_length.data.max().item()#以列表返回可遍历的(键, 值) 元组数组。针对字典
        encoder_hidden,encoder_outputs= self.encoder(ext_vocab_size,input_sentences,input_sentence_length)
        speakers_start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                                 input_conversation_length[:-1])), 0)  # cumsum当前位置之前该维度的值加和
        context=encoder_hidden.transpose(0,1)
        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]二维变三维
        speakers_masks = torch.stack(
            [pad(speakers.narrow(0, int(s / 2), int(l / 2)), int(max_len/2))  # narrow：取某一维度的几个值start开始长度为length
             for s, l in zip(speakers_start.data.tolist(),
                             input_conversation_length.data.tolist())], 0)
        speakers_masks=speakers_masks.transpose(0,1)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)#cumsum当前位置之前该维度的值加和

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]二维变三维
        context = torch.stack([pad(context.narrow(0, s, l), max_len) #narrow：取某一维度的几个值start开始长度为length
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        context=context.view(batch_size,int(max_len/2),2,-1)
        # utterance_c = context[:,:,1,:]
        # utterance_c=utterance_c.view(batch_size,int(max_len/2),-1)
        context=context.transpose(0,1)
        content,P_sen_emotions,R_sen_emotions,P_party_states,R_party_states = self.feature_extraction(context,speakers_masks)
        content = content.transpose(0, 1)
        content = torch.cat([content[i, :int(l / 2), :]
                             for i, l in enumerate(input_conversation_length.data)])

        P_sen_emotions = P_sen_emotions.transpose(0, 1)
        R_sen_emotions = R_sen_emotions.transpose(0, 1)
        P_state_emotions = torch.cat([P_sen_emotions[i, :int(l / 2), :]
                                      for i, l in enumerate(input_conversation_length.data)])
        R_state_emotions = torch.cat([R_sen_emotions[i, :int(l / 2), :]
                                      for i, l in enumerate(input_conversation_length.data)])

        R_emotions_prior,kl_div=self.emp_policy(P_sen_emotions,input_conversation_length,decode)
        P_emotions_pre = self.emo_classification(P_state_emotions)
        if self.config.R_classification==True:
            R_emotions_pre = self.emo_classification(R_state_emotions)
            R_emotions_prior_pre = self.dropout(R_emotions_prior)
            R_emotions_prior_pre = self.emo_classification(R_emotions_prior_pre)
            kl_div=torch.nn.functional.kl_div(torch.log_softmax(R_emotions_prior_pre,dim=-1),torch.softmax(R_emotions_pre,dim=-1))
        else:
            R_emotions_pre=None

        decoder_init = self.context2decoder(content)
        R_emotions_prior=self.context2decoder_emotion(R_emotions_prior)
        # L_states_init=self.context2decoder_state(L_states)
        # [num_layers, batch_size, hidden_size]
        # decoder_init=content
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.config.D_g)
        R_emotions_prior = R_emotions_prior.view(self.decoder.num_layers, -1, self.config.D_e)
        input_ids_tensor=input_sentences.view(-1,2,sen_len)
        input_ids_tensor=input_ids_tensor.narrow(1,0,1)
        input_ids_tensor=input_ids_tensor.view(-1,sen_len)
        encoder_outputs=encoder_outputs.view(-1,2,40,600)
        encoder_outputs=encoder_outputs.narrow(1,0,1)
        encoder_outputs=encoder_outputs.squeeze(1)
        if not decode:
            coverage_losses,decoder_outputs = self.decoder(target_sentences,
                                           R_emotions_prior,
                                           encoder_outputs,
                                           input_ids_tensor,
                                           ext_vocab_size,
                                           init_h=decoder_init,
                                           decode=decode)
            return coverage_losses, kl_div, decoder_outputs, (P_emotions_pre,R_emotions_pre)
        else:
            coverage_losses,prediction = self.decoder(target_sentences,
                                                        R_emotions_prior,
                                                        encoder_outputs,
                                                        input_ids_tensor,
                                                        ext_vocab_size,
                                                        init_h=decoder_init,
                                                        decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            # coverage_losses,prediction, final_score, length = self.decoder.beam_decode(R_emotions_prior,encoder_outputs,
            #                                                            input_ids_tensor,ext_vocab_size,
            #                                                            init_h=decoder_init,sample=self.config.sample,pointer=self.config.pointer)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return coverage_losses, prediction, (P_emotions_pre,R_emotions_pre)