from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import lj_model
from lj_model import MaskedNLLLoss
from layers import masked_cross_entropy,multi_loss,multi_loss_2,loss1_lm,loss2_emo,reward_lookahead
from utils import to_var, time_desc_decorator, TensorboardWriter, pad_and_pack, normal_kl_div, to_bow, bag_of_words_loss, normal_kl_div, embedding_metric
import os
from tqdm import tqdm
from math import isnan
import re
import random
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils.vocab import OOVDict
from torch.autograd import Variable
from utils.eval import calc_bleu
import layers
import nltk
from scipy.stats import pearsonr
# from LDA.classifier import LDA
# from LDA.loss import LinearDiscriminativeLoss
word2vec_path = "../datasets/GoogleNews-vectors-negative300.bin"
DIALOG_ACTIONS=["acknowleding","agreeing","consoling","encouraging","questioning","suggesting","sympathizing","wishing"]
EMO_LABELS=["joyful","excited","proud", "grateful","hopeful","surprised","confident","content","impressed","trusting","faithful","prepared",
 "caring","devastated","anticipating","sentimental","anxious","apprehensive","nostalgic", "lonely","embarrassed", "ashamed","guilty",
 "sad","jealous", "terrified","afraid","disappointed","angry","annoyed","disgusted","furious"]
action_dict={}
for i, act in enumerate(DIALOG_ACTIONS):
    action_dict[i] = act
emo_dict = {}
for i, emo in enumerate(EMO_LABELS):
    emo_dict[i] = emo
def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

class MyMlpLayer(nn.Module):
    def __init__(self,D_e, n_classes):
        super(MyMlpLayer,self).__init__()
        self.D_e=D_e
        self.n_classes=n_classes
        self.e_cell=nn.GRUCell(D_e,D_e)
        self.emotion_learn = layers.FeedForward(D_e,
                                                D_e,
                                                num_layers=3,
                                                hidden_size=D_e,
                                                activation="Tanh")

        self.gamma = nn.Parameter(torch.randn(D_e)).cuda()
        self.beta = nn.Parameter(torch.randn(D_e)).cuda()

        self.moving_mean = Variable(torch.zeros(D_e)).cuda()
        self.moving_var = Variable(torch.zeros(D_e)).cuda()
        self.dropout = nn.Dropout(p=0.65)
        # self.lda=LDA()
    def linear(self, x0,y,input_conversation_length):
        q=x0.transpose(0,1)
        e_ = torch.zeros(0).type(x0.type())  # batch, D_e type指float32？
        e = e_
        for q_ in q :
            e0 = torch.zeros(x0.size()[0], self.D_e).type(q.type()) if e_.size()[0] == 0 \
                else e_  # 前一时刻的e
            e_ = self.e_cell(q_, e0)
            e_ = self.dropout(e_)  # 只有用户的情感
            e=torch.cat([e, e_.unsqueeze(0)],0)
        emotions=e.transpose(0,1)
        P_state_emotions = torch.cat([emotions[i, :int(l / 2), :]
                                      for i, l in enumerate(input_conversation_length.data)])

        x1 = self.emotion_learn(P_state_emotions)
        x1 = torch.nn.functional.log_softmax(x1, dim=1)
        # x1=batch_norm_1d(x1, self.gamma, self.beta, True, self.moving_mean, self.moving_var)
        # x1=self.dropout(x1)
        x3=nn.Linear(self.D_e,int(self.D_e//2)).cuda()(x1)
        # x3 = torch.tanh(x2)
        x4 = nn.Linear(int(self.D_e//2), self.n_classes).cuda()(x3)
        # self.lda.fit(x4, y)
        # prob=self.lda.prob(x4)
        # loss = loss_obj(x1, y)
        prob=x4
        # prob=torch.nn.functional.oftmax(x4)
        return prob

    def call(self, inputs,y,input_conversation_length):
        return self.linear(inputs,y,input_conversation_length)

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class RLSolver(object):
    def __init__(self, config, train_data_loader, eval_data_loader,test_data_loader, is_train=True, model=None):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.test_data_loader=test_data_loader
        #self.vocab = vocab
        #self.oov_dict = oov_dict
        self.is_train = is_train
        self.model = model
        # self.emotion_classification= MyMlpLayer(config.D_e, config.n_classes)
        self.log_var_a = torch.zeros((1,), requires_grad=True)
        self.log_var_b = torch.zeros((1,), requires_grad=True)
    @time_desc_decorator('Build Graph')
    def build_generate(self, cuda=True):

        if self.model is None:
            self.model = getattr(lj_model, self.config.model)(self.config)  # 返回对象属性
            # total=sum([param.nelement()] for param in self.model.parameters)
            # print("Number of parameter: %.2fM" % (total/1e6))
            # orthogonal initialiation for hidden weights
            # input gate bias for GRUs
            if self.config.mode == 'train' and self.config.checkpoint is None:
                print('Parameter initiailization')
                for name, param in self.model.named_parameters():
                    if 'weight_hh' in name:
                        print('\t' + name)
                        nn.init.orthogonal_(param)  # 用（半）正交矩阵填充输入的张量或变量。？相当于初始化？

                    # bias_hh is concatenation of reset, input, new gates
                    # only set the input gate bias to 2.0
                    if 'bias_hh' in name:
                        print('\t' + name)
                        dim = int(param.size(0) / 3)
                        param.data[dim:2 * dim].fill_(2.0)

        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print('\t' + name + '\t', list(param.size()))

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint)

        if self.is_train:
            self.writer = TensorboardWriter(self.config.logdir)
            self.optimizer = self.config.optimizer(
                # filter(lambda p: p.requires_grad, chain(self.model.parameters(), self.emotion_classification.parameters(),self.log_var_a,self.log_var_b)),
                [p for p in self.model.parameters()] ,
                lr=self.config.lr )#,weight_decay=0.00002  ,weight_decay=self.config.weight_decay
            self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones= [4,5], gamma=0.3)
        print(
            f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')



    def save_model(self, epoch):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        self.model.load_state_dict(torch.load(checkpoint))

    def write_summary(self, epoch_i):
        epoch_loss = getattr(self, 'epoch_loss', None)
        if epoch_loss is not None:
            self.writer.update_loss(
                loss=epoch_loss,
                step_i=epoch_i + 1,
                name='train_loss')

        epoch_recon_loss = getattr(self, 'epoch_recon_loss', None)
        if epoch_recon_loss is not None:
            self.writer.update_loss(
                loss=epoch_recon_loss,
                step_i=epoch_i + 1,
                name='train_recon_loss')

        epoch_kl_div = getattr(self, 'epoch_kl_div', None)
        if epoch_kl_div is not None:
            self.writer.update_loss(
                loss=epoch_kl_div,
                step_i=epoch_i + 1,
                name='train_kl_div')

        kl_mult = getattr(self, 'kl_mult', None)
        if kl_mult is not None:
            self.writer.update_loss(
                loss=kl_mult,
                step_i=epoch_i + 1,
                name='kl_mult')

        epoch_bow_loss = getattr(self, 'epoch_bow_loss', None)
        if epoch_bow_loss is not None:
            self.writer.update_loss(
                loss=epoch_bow_loss,
                step_i=epoch_i + 1,
                name='bow_loss')

        validation_loss = getattr(self, 'validation_loss', None)
        if validation_loss is not None:
            self.writer.update_loss(
                loss=validation_loss,
                step_i=epoch_i + 1,
                name='validation_loss')

    def sent2id(self, sentences,oov_dict,target=False):
        """word => word id"""
        # [max_conversation_length, max_sentence_length]
        sentences_tokenid=[]
        for i,one in enumerate(sentences):
            sentences_tokenid.append(self.model.encoder.vocab.sent2id(one,oov_dict,i,target))
        return sentences_tokenid
    def get_dict(self,tokens, ngram, gdict=None):
        """
        get_dict
        """
        token_dict = {}
        if gdict is not None:
            token_dict = gdict
        tlen = len(tokens)
        for i in range(0, tlen - ngram + 1):
            ngram_token = "".join(tokens[i:(i + ngram)])
            if token_dict.get(ngram_token) is not None:
                token_dict[ngram_token] += 1
            else:
                token_dict[ngram_token] = 1
        return token_dict

    def calc_distinct_ngram(self,pair_list, ngram):
        """
        calc_distinct_ngram
        """
        ngram_total = 0.0
        ngram_distinct_count = 0.0
        pred_dict = {}
        for predict_tokens, _ in pair_list:
            self.get_dict(predict_tokens, ngram, pred_dict)
        for key, freq in pred_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
            # if freq == 1:
            #    ngram_distinct_count += freq
        return (ngram_distinct_count + 0.0001) / (ngram_total + 0.0001)

    def calc_distinct(self,pair_list):
        """
        calc_distinct
        """
        distinct1 = self.calc_distinct_ngram(pair_list, 1)
        distinct2 = self.calc_distinct_ngram(pair_list, 2)
        return [distinct1, distinct2]
    def unpad(self,list):
        s=[]
        for a in list:
            if a=='<eos>':
                break
            else :
                s.append(a)
        str=' '.join(s)
        return str
    @time_desc_decorator('Training Start!')
    def train(self):
        epoch_loss_history = []
        epoch_emo_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_lm_history = []
            batch_loss_emo_history = []
            self.model.train()
            total=sum([param.nelement() for param in self.model.parameters()])
            print("Number of parameter :%2fM" % (total/1e6))
            for batch_i, (sentences, sentence_length, conversation_length,conversation_turns, speakers, emotion_labels) \
                    in enumerate(tqdm(self.train_data_loader, ncols=32)):
                oov_dict = OOVDict(self.config.vocab_size)
                speakers = [s for s in speakers]
                input_sentences = [sent for conv in sentences for turn in conv for sent in turn]
                input_sentences = self.sent2id(input_sentences, oov_dict,target=False)
                target_sentences = [turn[1] for conv in sentences for turn in conv]
                target_sentences=self.sent2id(target_sentences, oov_dict,target=True)
                input_sentence_length = [l for len_list in sentence_length for turn_l in len_list for l in turn_l]
                input_conversation_length = [l for l in conversation_length]
                P_target_emotions=[]
                R_target_emotions = []
                for emo in emotion_labels:
                    for i,e in enumerate(emo):
                        if i%2==0:
                            P_target_emotions.append(int(e))
                        else:
                            R_target_emotions.append(int(e))
                # target_emotions=[int(emo[1]) for emo in emotion_labels ]
                speakers_mask = [s for one in speakers for s in one]
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_conversation_turns=to_var(torch.LongTensor(conversation_turns))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                speakers_mask = to_var(torch.FloatTensor(speakers_mask))
                P_target_emotions=to_var(torch.LongTensor(P_target_emotions))
                R_target_emotions = to_var(torch.LongTensor(R_target_emotions))
                # reset gradient
                coverage_losses,kl_div,sentence_logits,emotions_pred= self.model(
                    self.config.batch_size,
                    oov_dict.ext_vocab_size,
                    input_sentences,
                    input_sentence_length,
                    input_conversation_turns,
                    input_conversation_length,
                    target_sentences,
                    speakers_mask,
                    decode=False)
                P_emotions_pred,R_emotions_pred=emotions_pred
                log_prob_sentence=torch.nn.functional.log_softmax(sentence_logits,2)
                P_log_prob_emotions = torch.nn.functional.log_softmax(P_emotions_pred,1)
                # R_log_prob_emotion = torch.nn.functional.log_softmax(R_emotions_pred, 1)
                batch_loss, batch_loss_lm, batch_loss_emo_class ,ppl= multi_loss(log_prob_sentence, P_log_prob_emotions,
                                                                               target_sentences, P_target_emotions,
                                                                               conversation_length,sentence_length)
                batch_loss=batch_loss_lm+0.5*batch_loss_emo_class+coverage_losses#+kl_div.mean()
                if self.config.R_classification==True:
                    R_log_prob_emotions = torch.nn.functional.log_softmax(R_emotions_pred, 1)
                    R_log_prob_emotions_flat = R_log_prob_emotions.contiguous().view(-1, R_log_prob_emotions.size(-1))
                    R_target_emotions_flat = R_target_emotions.contiguous().view(-1)
                    batch_loss_emotions_R = torch.nn.NLLLoss()(R_log_prob_emotions_flat,R_target_emotions_flat)
                    batch_loss+=0.5*(batch_loss_emotions_R)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                #reward function model
                assert not isnan(batch_loss_lm.item())  # （not a number 返回true，否则返回false# ） assert= if not 如果不是个数则触发异常
                # n_total_emotions += log_prob_emotion.size()[0]
                if batch_i % self.config.print_every == 0:
                    epoch_str=f''
                    if self.config.R_classification==True:
                        epoch_str=epoch_str+f'Remoloss = {batch_loss_emotions_R.item() :.3f},'
                    tqdm.write(
                        f'Epoch: {epoch_i + 1}, iter {batch_i}: lmloss = {batch_loss_lm.item() :.3f},coverage_losses={coverage_losses.item():.3f}, emoloss = {batch_loss_emo_class.item() :.3f} ,kl_div = {kl_div:.3f},'+epoch_str)  # 写日志reward = {reward.item():.3f},rlemo_class_loss = {rl_batch_loss_emo.item() :.3f}  reward = {reward.item():.3f},rlemo_class_loss = {rl_batch_loss_emo.item() :.3f}, , emoloss = {batch_loss_emo_class.item() :.3f} , emoloss = {batch_loss_emo_class.item() :.3f} v ,rlemo_class_loss = {rl_batch_loss_emo.item() :.3f},reward = {reward:.3f}
                # batch_loss = 0.5 * batch_loss_lm + 0.5 * batch_loss_emo_class
                batch_loss_lm_history.append(batch_loss_lm.item())
                batch_loss_emo_history.append(batch_loss_emo_class.item())
            epoch_loss = np.mean(batch_loss_lm_history) #+
            epoch_emo_loss= np.mean(batch_loss_emo_history)
            self.epoch_loss = epoch_loss

            print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f},{epoch_emo_loss:.3f}'
            print(print_str)

            if epoch_i % self.config.save_every_epoch == 0:
                self.save_model(epoch_i + 1)
                # self.save_emo_model(epoch_i + 1)
            print('\n<Validation>...')
            self.validation_loss = self.evaluate()#调用后面的验证方法

            if epoch_i % self.config.plot_every_epoch == 0:
                self.write_summary(epoch_i)

            # torch.cuda.empty_cache()
            self.scheduler.step()
        self.save_model(self.config.n_epoch)
        # self.save_emo_model(self.config.n_epoch)

        return epoch_loss_history

    def generate_sentence(self,batch_size, oov_dict,input_sentences_token,target_sentences_token, input_sentences, input_sentence_length,
                          input_conversation_turns, input_conversation_length, target_sentences, target_emotions,
                          speakers_mask):
        self.model.eval()
        # self.emotion_classification.eval()
        smooth = SmoothingFunction()
        # [batch_size, max_seq_len, vocab_size]
        coverage_losses, generated_sentences, emotions_pred= self.model(
            batch_size,
            oov_dict.ext_vocab_size,
            input_sentences,
            input_sentence_length,
            input_conversation_turns,
            input_conversation_length,
            target_sentences,
            speakers_mask,
            decode=True)
        num_sentences = input_sentences.size(0)
        input_sentences_token = np.reshape(input_sentences_token,[int(num_sentences / 2), 2, self.config.max_unroll])
        input_sentences_token = input_sentences_token[:,0,:]
        P_emotions_pred, R_emotions_pred = emotions_pred
        P_emotions_target, R_emotions_target = target_emotions
        with open(os.path.join(self.config.save_path, 'samples.txt'), 'a') as f:
            # f.write(f'<Epoch {self.epoch_i}>\n\n')
            belu_scores = []
            sents = []
            P_emo_pre = torch.argmax(P_emotions_pred, dim=1)
            P_emo_acc = torch.sum(P_emotions_target == P_emo_pre).data
            if self.config.R_classification==True:
                R_emo_pre = torch.argmax(R_emotions_pred, dim=1)
                R_emo_acc = torch.sum(R_emotions_target == R_emo_pre).data
            else:
                R_emo_acc=0
            # tqdm.write('\n<Samples>')
            for i, content in enumerate(zip(input_sentences_token, target_sentences_token, generated_sentences,P_emotions_target,R_emotions_target,P_emo_pre,R_emo_pre)):
                input_sent = self.unpad(content[0])
                target_sent = self.unpad(content[1])
                f.write("post_emotion_state:"+emo_dict[content[3].item()]+'\n')
                f.write(input_sent + '\n')
                f.write(target_sent + '\n')
                target_sent = target_sent.strip().split(' ')
                beam_score = []
                beam_sentence = []
                sent=content[2]
                output_sent = self.model.encoder.vocab.decode(i, sent, oov_dict)
                beam_sentence.append(output_sent)
                output_sent = nltk.word_tokenize(output_sent.strip())
                    # print(output_sent,'\n',[target_sent])
                score = sentence_bleu([target_sent], output_sent, weights=(0.25, 0.25, 0.25, 0.25),
                                          smoothing_function=smooth.method1)
                beam_score.append(score)
                sent = [output_sent, target_sent]
                sents.append(sent)
                max_index = beam_score.index(max(beam_score))
                belu_scores.append(beam_score[max_index])
                output_sent = beam_sentence[max_index]
                f.write(output_sent + '\n')
                f.write("post_pre_emotion_state:"+emo_dict[content[5].item()]+'\n')
                f.write("re_emotion_state:" + emo_dict[content[4].item()] + '\n')
                f.write("re_pre_emotion_state:"+emo_dict[content[6].item()]+'\n')
                # f.write(str(multiturn)+'\n')
                f.write('\n')
            # distinct1, distinct2 = self.calc_distinct(sents)
            belu_score = np.sum(belu_scores) / len(belu_scores)
            # print("BELU",belu_score)
            # print("accuracy", accuracy)
            # print('')
            return belu_score, sents, P_emo_acc,R_emo_acc

    def evaluate(self):
        self.model.eval()
        # self.emotion_classification.eval()
        batch_loss_lm_history = []
        batch_loss_emo_history_P = []
        ppl_scores=[]
        for batch_i, (
        sentences, sentence_length, conversation_length, conversation_turns, speakers, emotion_labels) in enumerate(
                tqdm(self.eval_data_loader, ncols=32)):
            oov_dict = OOVDict(self.config.vocab_size)
            speakers = [s for s in speakers]
            input_sentences_token = [sent for conv in sentences for turn in conv for sent in turn]
            input_sentences = self.sent2id(input_sentences_token, oov_dict, target=False)
            target_sentences_token = [turn[1] for conv in sentences for turn in conv]
            target_sentences = self.sent2id(target_sentences_token, oov_dict, target=True)
            input_sentence_length = [l for len_list in sentence_length for turn_l in len_list for l in turn_l]
            input_conversation_length = [l for l in conversation_length]
            P_target_emotions = []
            P_target_behaviors = []
            R_target_emotions = []
            R_target_behaviors = []
            for emo in emotion_labels:
                for i, e in enumerate(emo):
                    if i % 2 == 0:
                        P_target_emotions.append(int(e))
                    else:
                        R_target_emotions.append(int(e))
            speakers_mask = [s for one in speakers for s in one]
            with torch.no_grad():
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                input_conversation_turns = to_var(torch.LongTensor(conversation_turns))
                speakers_mask = to_var(torch.FloatTensor(speakers_mask))
                P_target_emotions = to_var(torch.LongTensor(P_target_emotions))
                R_target_emotions = to_var(torch.LongTensor(R_target_emotions))

            if self.epoch_i>=0:
                coverage_losses, kl_div, behavior_kl_div, sentence_logits, emotions_pred = self.model(
                    self.config.eval_batch_size,
                    oov_dict.ext_vocab_size,
                    input_sentences,
                    input_sentence_length,
                    input_conversation_turns,
                    input_conversation_length,
                    target_sentences,
                    speakers_mask,
                    decode=False)  #
                P_emotions_pred, R_emotions_pred = emotions_pred
                log_prob_emotion = torch.nn.functional.log_softmax(P_emotions_pred, 1)
                log_prob_sentence = torch.nn.functional.log_softmax(sentence_logits, 2)
                batch_loss, batch_loss_lm, batch_loss_emo_class,ppl = multi_loss(log_prob_sentence, log_prob_emotion,
                                                                         target_sentences, P_target_emotions,
                                                                         conversation_length, sentence_length)
                batch_loss_lm_history.append(batch_loss_lm.item())
                batch_loss_emo_history_P.append(batch_loss_emo_class.item())
                ppl_scores.append(ppl)
        epoch_ppl=np.mean(ppl_scores)
        epoch_loss = np.mean(batch_loss_lm_history)  # +
        epoch_loss_emo_P = np.mean(batch_loss_emo_history_P)
        #
        # distinct1,distinct2=self.calc_distinct(all_sentences)
        # bleu1, bleu2, bleu3, bleu4, avg = calc_bleu(all_sentences)
        print_str=f''
        print_str += f'Validation loss: {epoch_loss:.3f} ,{epoch_loss_emo_P:.3f},'#{epoch_loss_behavior_P:.3f},

        print(print_str)

    def test(self):
        self.model.eval()
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter :%2fM" % (total / 1e6))
        # self.emotion_classification.eval()
        batch_loss_emo_history = []
        batch_loss_lm_history = []
        batch_loss_behavior_history = []
        batch_bleu_scores = []
        ppl_scores=[]
        all_emo_num=0
        P_true_pre_emo = 0
        R_true_pre_emo = 0
        all_sentences=[]
        for batch_i, (
                sentences, sentence_length, conversation_length, conversation_turns, speakers,
                emotion_labels) in enumerate(tqdm(self.test_data_loader, ncols=32)):
            oov_dict = OOVDict(self.config.vocab_size)
            speakers = [s for s in speakers]
            input_sentences_token = [sent for conv in sentences for turn in conv for sent in turn]
            input_sentences = self.sent2id(input_sentences_token, oov_dict, target=False)
            target_sentences_token = [turn[1] for conv in sentences for turn in conv]
            target_sentences = self.sent2id(target_sentences_token, oov_dict, target=True)
            input_sentence_length = [l for len_list in sentence_length for turn_l in len_list for l in turn_l]
            # target_sentence_length = [turn_l[1] for len_list in sentence_length for turn_l in len_list]
            input_conversation_length = [l for l in conversation_length]
            input_conversation_turns = [l for l in conversation_turns]
            P_target_emotions = []
            P_target_behaviors = []
            R_target_emotions = []
            R_target_behaviors = []
            for emo in emotion_labels:
                for e in enumerate(emo):
                    if i % 2 == 0:
                        P_target_emotions.append(int(e))
                    else:
                        R_target_emotions.append(int(e))
            speakers_mask = [s for one in speakers for s in one]
            with torch.no_grad():
                input_sentences = to_var(torch.LongTensor(input_sentences))
                target_sentences = to_var(torch.LongTensor(target_sentences))
                input_sentence_length = to_var(torch.LongTensor(input_sentence_length))
                input_conversation_turns= to_var(torch.LongTensor(input_conversation_turns))
                input_conversation_length = to_var(torch.LongTensor(input_conversation_length))
                speakers_mask = to_var(torch.FloatTensor(speakers_mask))
                P_target_emotions = to_var(torch.LongTensor(P_target_emotions))
                R_target_emotions = to_var(torch.LongTensor(R_target_emotions))
            bleu, sents,P_emo_acc,R_emo_acc = self.generate_sentence(self.config.eval_batch_size,
                                                                     oov_dict,
                                                                     input_sentences_token,
                                                                     target_sentences_token,
                                                                     input_sentences,
                                                                     input_sentence_length,
                                                                     input_conversation_turns,
                                                                     input_conversation_length,
                                                                     target_sentences,
                                                                       (P_target_emotions, R_target_emotions),
                                                                       speakers_mask)
            batch_bleu_scores.append(bleu)
            all_emo_num += len(P_target_emotions)
            P_true_pre_emo += P_emo_acc
            R_true_pre_emo += R_emo_acc
            all_sentences += sents
            coverage_losses, kl_div, behavior_kl_div, sentence_logits, emotions_pred= self.model(
                self.config.eval_batch_size,
                oov_dict.ext_vocab_size,
                input_sentences,
                input_sentence_length,
                input_conversation_turns,
                input_conversation_length,
                target_sentences,
                speakers_mask,
                decode=False)
            P_emotions_pred, R_emotions_pred = emotions_pred
            log_prob_emotion = torch.nn.functional.log_softmax(P_emotions_pred, 1)
            log_prob_sentence = torch.nn.functional.log_softmax(sentence_logits, 2)
            batch_loss, batch_loss_lm, batch_loss_emo_class,ppl = multi_loss(log_prob_sentence, log_prob_emotion,
                                                                         target_sentences, P_target_emotions,
                                                                         conversation_length, sentence_length,
                                                                         )

            ppl_scores.append(ppl)
            batch_loss_lm_history.append(batch_loss_lm.item())
            batch_loss_emo_history.append(batch_loss_emo_class.item())
        # epoch_ppl=np.mean(ppl_scores)
        epoch_loss = np.mean(batch_loss_lm_history)  # +
        epoch_loss_emo = np.mean(batch_loss_emo_history)

        bleu1, bleu2, bleu3, bleu4, avg = calc_bleu(all_sentences)
        # bleu = np.mean(batch_bleu_scores)
        P_emo_acc = P_true_pre_emo / all_emo_num
        R_emo_acc = R_true_pre_emo / all_emo_num


        ppl=np.exp(epoch_loss)
        distinct1, distinct2 = self.calc_distinct(all_sentences)
        print_str = f'Validation loss: {epoch_loss:.3f} ,{epoch_loss_emo:.3f}\n'
        print(print_str)
        print("BLEU-4",bleu4,"BLEU-AVG",avg,  "distinct1", distinct1, "distinct2", distinct2,"PPL",ppl)
        print("P_emo", P_emo_acc,"R_emo", R_emo_acc)#
