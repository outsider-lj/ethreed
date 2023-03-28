"""Library of functions for calculating rewards
Note that rewards should be normalized for best results.
"""
import os
import string
import pickle
from pathlib import Path

import numpy as np
import gensim
from nltk.corpus import stopwords


from model.utils import embedding_metric, Tokenizer

EPSILON = np.finfo(np.float32).eps
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tokenizer = Tokenizer('spacy')
# stopwords = stopwords.words('english')
# question_words = {'who', 'what', 'why', 'where', 'how', 'when'}
# _ = [stopwords.remove(q) for q in question_words]
# punct = list(string.punctuation)
# contractions = ["'s", "'d", "'ld", "n't", "'re", "'ll", "'ve"]
# filters = set(stopwords + contractions + punct)


def _get_emojis():
    # All emojis in the order returned by deepmoji
    EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: :pensive: " + \
             ":ok_hand: :blush: :heart: :smirk: :grin: :notes: :flushed: " + \
             ":100: :sleeping: :relieved: :relaxed: :raised_hands: " + \
             ":two_hearts: :expressionless: :sweat_smile: :pray: " + \
             ":confused: :kissing_heart: :heartbeat: :neutral_face: " + \
             ":information_desk_person: :disappointed: :see_no_evil: " + \
             ":tired_face: :v: :sunglasses: :rage: :thumbsup: :cry: " + \
             ":sleepy: :yum: :triumph: :hand: :mask: :clap: :eyes: :gun: " + \
             ":persevere: :smiling_imp: :sweat: :broken_heart: " + \
             ":yellow_heart: :musical_note: :speak_no_evil: :wink: :skull: " + \
             ":confounded: :smile: :stuck_out_tongue_winking_eye: :angry: " + \
             ":no_good: :muscle: :facepunch: :purple_heart: " + \
             ":sparkling_heart: :blue_heart: :grimacing: :sparkles:"
    EMOJIS = EMOJIS.split(' ')
    return EMOJIS


def _get_emojis_to_rewards_dict():
    # How detected emojis map to rewards
    emojis_to_rewards = {
        # very strongly positive
        ':kissing_heart:': 1, ':thumbsup:': 1, ':ok_hand:': 1,
        ':smile:': 1,

        # strongly positive
        ':blush:': 0.75, ':wink:': 0.75, ':muscle:': 0.75,
        ':grin:': 0.75, ':heart_eyes:': 0.75, ':100:': 0.75,

        # positive
        ':smirk:': 0.5, ':stuck_out_tongue_winking_eye:': 0.5,
        ':sunglasses:': 0.5, ':relieved:': 0.5, ':relaxed:': 0.5,
        ':blue_heart:': 0.5, ':two_hearts:': 0.5, ':heartbeat:': 0.5,
        ':yellow_heart:': 0.5,

        # negative
        ':disappointed:': -0.5, ':eyes:': -0.5,
        ':expressionless:': -0.5, ':sleeping:': -0.5,
        ':grimacing:': -0.5,

        # strongly negative
        ':neutral_face:': -0.75, ':confused:': -0.75,
        ':triumph:': -0.75, ':confounded:': -0.75,

        # very strongly negative
        ':unamused:': -1, ':angry:': -1,  # removing ':hand:': -1 due to ambiguity
        ':rage:': -1
    }
    return emojis_to_rewards


def _get_reward_multiplier():
    EMOJIS = _get_emojis()
    emojis_to_rewards = _get_emojis_to_rewards_dict()
    reward_multiplier = np.zeros(len(EMOJIS))
    for emoji, reward_val in emojis_to_rewards.items():
        loc = EMOJIS.index(emoji)
        reward_multiplier[loc] = reward_val
    return reward_multiplier


def normalizeZ(x):
    x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std + EPSILON)

def pad(a,length):
    if length >len(a):
            return np.concatenate([a,
                              np.zeros(length - len(a))])#原来有.cuda()方法
    else:
        return a


def discount(rewards, conv_len,gamma=0.9):
    """Convert raw rewards from batch of episodes to discounted rewards.
    Args:
        rewards: [batch_size, episode_len]
    Returns:
        discounted: [batch_size, episode_len]
    """

    batch_size = conv_len.shape[0]
    episode_len =np.max(conv_len.data.tolist()) //2
    start = np.cumsum(np.concatenate((np.zeros([1]),
                                      conv_len.data.tolist()[:-1])), 0)  # cumsum当前位置之前该维度的值加和

    # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]二维变三维
    rewards = np.stack([pad(rewards[int(s//2):int(s//2+l//2)], episode_len)  # narrow：取某一维度的几个值start开始长度为length
                                  for s, l in zip(start,
                                                  conv_len.data.tolist())], 0)
    discounted = np.zeros([batch_size,episode_len])
    running_add = np.zeros((batch_size))
    for step in reversed(range(episode_len)):
        running_add = gamma * running_add + rewards[:, step]
        discounted[:, step] = running_add
    return discounted

