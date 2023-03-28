import torch
from torch.nn import functional as F
import torch.nn as nn
from utils import to_var, sequence_mask
from utils.vocab import PAD_ID,UNK_ID
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math
import numpy as np
#1.只有MLE(一个文本生成loss)masked_cross_entropy
#2.MLE+情感识别的损失 multi_loss(直接相加)multi_loss2借助方差两个损失均匀下降
#3.MLE+分层强化学习损失（manager——情感识别，worker——单词生成）
def reward_lookahead(emotions, conv_length):
    # convert = [1,1,-1,-1, -1]
    # convert = to_var(torch.FloatTensor(convert))
    sum = 0
    reward_list = torch.zeros(emotions.size()[0]).type(emotions.type())
    for len in conv_length:
        for i in range(1, int(len  / 2)+1):
            if i == int(len / 2):
                reward_list[sum] = torch.tensor(-0.0)
                sum = sum + 1
                break
            emo_now = emotions[sum]
            emo_next = emotions[sum + 1]
            reward =(emo_next-emo_now)*10
            reward_list[sum]=-(reward[0]-reward[1])
            sum = sum + 1
    return reward_list
def loss2_emo(cos,log_p_z, conv_len, per_example=False):
    cos = cos.data.tolist()
    discounted = hrl_rewards.discount(cos, conv_len)
    # reward = hrl_rewards.normalizeZ(discounted)
    reward = torch.tensor(discounted).cuda()
    reward = torch.cat([reward[i, :int(l / 2)]
                        for i, l in enumerate(conv_len.data)])
    # batch_size,  num_classes = logits.size()
    # log_probs_flat = F.log_softmax(logits, dim=1)
    # probs_flat=F.softmax(logits,dim=1)
    # target_flat=target.view(-1,1)
    # losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # reward=reward2_lookahead(probs_flat, conv_len)
    losses=reward*(log_p_z)
    return losses.mean()


def loss1_lm(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    smooth = SmoothingFunction()
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    generate=torch.max(log_probs_flat,dim=1)
    # [batch_size * max_len, 1]
    generate_idx_flat=generate[1].view(-1, 1)
    # generate_idx_flat = -torch.gather(log_probs_flat, dim=1, index=generate_flat)
    # 句子级别
    target_idx = target.view(batch_size, max_len)
    generate_idx = generate_idx_flat.view(batch_size, max_len)
    # equal=target_idx.eq(generate_idx)
    # reward=torch.sum(equal,dim=1)
    beam_score=[]
    for target_sent,output_sent,len in zip(target_idx,generate_idx,length):
        target_sent=[str(target_sent[i].item()) for i in range(len)]
        output_sent = [str(output_sent[i].item()) for i in range(len)]
        score = sentence_bleu([target_sent], output_sent, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smooth.method1)
        beam_score.append(score*100+0.01)
    reward=torch.tensor(beam_score).cuda()
   #求mle——loss_lm
    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(batch_size, max_len)
    mask = sequence_mask(sequence_length=length, max_len=max_len)
    losses = losses * mask.float()
    losses=losses.sum(1)
    losses=losses/length.float()
    loss=reward*losses
    return loss.mean()


def hrl_loss(s,e,target_s,target_e, conv_l,sen_l,log_var_a,log_var_b):
    losses = []
    losses_lm = []
    losses_emo = []
    roll = target_s.size(0)
    target_e_vector=torch.zeros([roll,5]).cuda()
    for i,e_index in enumerate(target_e):
        target_e_vector[i][e_index]=1.0

    R_emo_manager=torch.exp(-torch.cosine_similarity(e,target_e_vector,dim=1))
    R_lm_worker=reward2_lookahead(e,conv_l)
    sen_l=[l[0] for len in sen_l for l in len[1:] ]
    for i in range(roll):
        p_emo, r_emo ,reward_emo= e[i], target_e[i],R_emo_manager[i]
        p_sent, r_sent,len,reward_lm = s[i],target_s[i],sen_l[i],R_lm_worker[i]
        loss_lm = criterion(torch.log(p_sent), r_sent, log_var_a, len)
        loss_emo = criterion(torch.log(p_emo.unsqueeze(0)), r_emo.unsqueeze(0), log_var_b, 1)
        # loss_lm = reward_emo*nn.NLLLoss(ignore_index=PAD_ID)(torch.log(p_sent), r_sent)
        # loss_emo=reward_emo*nn.NLLLoss(ignore_index=PAD_ID)(torch.log(p_emo.unsqueeze(0)), r_emo.unsqueeze(0))
        loss = reward_emo*loss_lm +reward_emo*loss_emo
        # loss = loss / 2
        losses_lm.append(loss_lm)
        losses_emo.append(loss_emo)
        losses.append(loss)

    loss_total = losses[0]
    loss_total_lm = losses_lm[0]
    loss_total_emo = losses_emo[0]
    for i in range(1, roll):
        loss_total = loss_total + losses[i]
        loss_total_lm = loss_total_lm + losses_lm[i]
        loss_total_emo = loss_total_emo + losses_emo[i]

    loss = loss_total / roll
    loss_lm = loss_total_lm / roll
    loss_emo = loss_total_emo / roll

    return loss, loss_lm, loss_emo
def criterion_1(y_pred, y_true, log_vars,length):
  loss = 0

  for i in range(length):
    n=y_pred.size(1)
    y_true_convert = torch.zeros(n)
    precision = torch.exp(-log_vars)
    if y_true[i]>=n:
        y_true_convert[UNK_ID]=1.0
    else:
        y_true_convert[y_true[i]]=1.0
    # a=y_pred[i]-y_true_convert
    diff = (y_pred[i]-y_true_convert)**2
    # print(precision * diff + log_vars)
    loss += torch.sum(precision * diff+ log_vars,-1)
    # if loss > 1:
    #     print((y_pred-y_true_convert))
  return loss/length

def criterion(y_pred, y_true, log_vars,length):
    loss = 0
    max_len, num_classes = y_pred.size()
    target_flat = y_true.view(-1, 1)
    losses_flat = -torch.gather(y_pred, dim=1, index=target_flat)
    precision = torch.exp(-log_vars)
    losses_flat=losses_flat.narrow(0,0,int(length))
    loss=torch.sum(precision * losses_flat+ log_vars+torch.log(torch.FloatTensor([2]).cuda()),-1)
    # if loss > 1:
    #     print((y_pred-y_true_convert))
    return torch.mean(loss)

def multi_loss_2(s,e,target_s,target_e, conv_l,sen_l,log_var_a,log_var_b):
    losses = []
    losses_lm=[]
    losses_emo=[]

    roll=target_s.size(0)
    # conv_l=[c[0] for ]
    sen_l=[l[1] for len in sen_l for l in len ]
    for i in range(roll):
        p_emo, r_emo = e[i], target_e[i]
        p_sent, r_sent,len = s[i],target_s[i],sen_l[i]

        loss_lm = criterion(p_sent, r_sent,log_var_a,len)
        loss_emo= criterion(p_emo.unsqueeze(0), r_emo.unsqueeze(0),log_var_b,1)
        loss = loss_lm  + loss_emo
        # loss = loss / 2
        losses_lm.append(loss_lm)
        losses_emo.append(loss_emo)
        losses.append(loss)

    loss_total = losses[0]
    loss_total_lm=losses_lm[0]
    loss_total_emo=losses_emo[0]
    for i in range(1, roll):
        loss_total = loss_total+losses[i]
        loss_total_lm =loss_total_lm+ losses_lm[i]
        loss_total_emo =loss_total_emo+ losses_emo[i]

    loss = loss_total / roll
    loss_lm=loss_total_lm/roll
    loss_emo=loss_total_emo/roll

    return loss,loss_lm,loss_emo

def multi_loss(s,e,target_s,target_e, conv_l,sen_l):
    losses = []
    losses_lm=[]
    losses_emo=[]
    roll=target_s.size(0)
    # conv_l=[c[0] for ]
    sen_l=[l[0] for len in sen_l for l in len[1:] ]
    for i in range(roll):
        p_emo, r_emo = e[i], target_e[i]
        p_sent, r_sent,len = s[i],target_s[i],sen_l[i]

        loss_lm = nn.CrossEntropyLoss(ignore_index=PAD_ID)(p_sent, r_sent)
        loss_emo=nn.NLLLoss()(p_emo.unsqueeze(0), r_emo.unsqueeze(0))
        loss = loss_lm * 0.7 + loss_emo * 0.3
        loss = loss / 2
        losses_lm.append(loss_lm)
        losses_emo.append(loss_emo)
        losses.append(loss)

    loss_total = losses[0]
    loss_total_lm=losses_lm[0]
    loss_total_emo=losses_emo[0]
    for i in range(1, roll):
        loss_total += losses[i]
        loss_total_lm += losses_lm[i]
        loss_total_emo += losses_emo[i]

    loss = loss_total / roll
    loss_lm=loss_total_lm/roll
    loss_emo=loss_total_emo/roll

    return loss,loss_lm,loss_emo

def multi_loss(s,e,target_s,target_e, conv_l,sen_l):
    losses = []
    losses_lm=[]
    losses_emo=[]
    ppl_scores=[]
    roll=target_s.size(0)
    # conv_l=[c[0] for ]
    sen_l=[l[1] for len in sen_l for l in len ]
    for i in range(roll):
        p_emo, r_emo = e[i], target_e[i]
        p_sent, r_sent,len = s[i],target_s[i],sen_l[i]

        loss_lm = nn.NLLLoss(ignore_index=PAD_ID)(p_sent, r_sent)
        loss_emo=nn.NLLLoss()(p_emo.unsqueeze(0), r_emo.unsqueeze(0))
        loss = loss_lm + loss_emo
        ppl=math.exp(loss_lm)
        ppl_scores.append(ppl)
        losses_lm.append(loss_lm)
        losses_emo.append(loss_emo)
        losses.append(loss)

    loss_total = losses[0]
    loss_total_lm=losses_lm[0]
    loss_total_emo=losses_emo[0]
    for i in range(1, roll):
        loss_total += losses[i]
        loss_total_lm += losses_lm[i]
        loss_total_emo += losses_emo[i]

    loss = loss_total / roll
    loss_lm=loss_total_lm/roll
    loss_emo=loss_total_emo/roll
    ppl=np.mean(ppl_scores)
    return loss,loss_lm,loss_emo,ppl
# https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def masked_cross_entropy(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)

    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)
    #losses=torch.nn.NLLLoss()(losses,target)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    loss = losses.sum() / length.float().sum()

    if per_example:
        # loss: [batch_size]
        return losses.sum(1)
    else:
        loss = losses.sum()
        return loss, length.float().sum()

def reward3_wordcover(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    logits_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    generate=torch.max(log_probs_flat,dim=1)
    # [batch_size * max_len, 1]
    generate_flat=generate[1].view(-1, 1)
    generate_idx_flat = -torch.gather(log_probs_flat, dim=1, index=generate_flat)
    # 句子级别
    target_idx = target.view(batch_size, max_len)
    generate_idx = generate_idx_flat.view(batch_size, max_len)
    equal=target_idx.eq(generate_idx)
    reward=torch.sum(equal,dim=1)
    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    # losses_flat = -torch.gather(log_probs_flat, dim=1, index=generate_flat)
    #

   #求mle——loss_lm
    # [batch_size * max_len, 1]
    target_flat = target.view(-1, 1)

    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)
    # losses=torch.nn.NLLLoss()(losses,target)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()
    losses.sum(1)
    loss=reward*losses
    return loss.sum()

def masked_cross_entropy(logits, target, length, per_example=False):
    """
    Args:
        logits (Variable, FloatTensor): [batch, max_len, num_classes]
            - unnormalized probability for each class
        target (Variable, LongTensor): [batch, max_len]
            - index of true class for each corresponding step
        length (Variable, LongTensor): [batch]
            - length of each data in a batch
    Returns:
        loss (Variable): []
            - An average loss value masked by the length
    """
    batch_size, max_len, num_classes = logits.size()

    # [batch_size * max_len, num_classes]
    log_probs_flat = logits.view(-1, num_classes)

    # [batch_size * max_len, num_classes]


    # [batch_size * max_len, 1]
    #
    _,gen=torch.max(log_probs_flat,dim=1)
    # Negative Log-likelihood: -sum {  1* log P(target)  + 0 log P(non-target)} = -sum( log P(target) )
    # [batch_size * max_len, 1]
    target_flat = gen.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # [batch_size, max_len]
    losses = losses_flat.view(batch_size, max_len)
    #losses=torch.nn.NLLLoss()(losses,target)

    # [batch_size, max_len]
    mask = sequence_mask(sequence_length=length, max_len=max_len)

    # Apply masking on loss
    losses = losses * mask.float()

    # word-wise cross entropy
    return losses.sum(1)
