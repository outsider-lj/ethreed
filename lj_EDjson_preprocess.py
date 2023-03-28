# Preprocess cornell movie dialogs dataset
import json
import os
from multiprocessing import Pool
import argparse
import pickle
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model.utils import Tokenizer, Vocab, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import pandas as pd
project_dir = Path(__file__).resolve().parent
print(project_dir)
datasets_dir = project_dir.joinpath('datasets/ED/')
tokenizer = Tokenizer('spacy')
DIALOG_ACTIONS=["acknowleding","agreeing","consoling","encouraging","questioning","suggesting","sympathizing","wishing"]
EMO_LABELS=["joyful","excited","proud", "grateful","hopeful","surprised","confident","content","impressed","trusting","faithful","prepared",
 "caring","devastated","anticipating","sentimental","anxious","apprehensive","nostalgic", "lonely","embarrassed", "ashamed","guilty",
 "sad","jealous", "terrified","afraid","disappointed","angry","annoyed","disgusted","furious"]

def clean(sentence):
    word_pairs1 = {" thats ": " that's ", " dont ": " don't ", " doesnt ": " doesn't ", " didnt ": " didn't ",
                   " youd ": " you'd ",
                   " youre ": " you're ", " youll ": " you'll ", " im ": " i'm ", " theyre ": " they're ",
                   " whats ": "what's", " couldnt ": " couldn't ", " souldnt ": " souldn't ", " ive ": " i've ",
                   " cant ": " can't ", " arent ": " aren't ", " isnt ": " isn't ", " wasnt ": " wasn't ",
                   " werent ": " weren't ", " wont ": " won't ", " theres ": " there's ", " therere ": " there're "," its ": " it's "}
    word_pairs2 = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                   "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                   "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have",
                   "can't": "cannot",
                   "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                   "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}
    for k, v in word_pairs1.items():
        sentence = sentence.replace(k, v)
    for k, v in word_pairs2.items():
        sentence = sentence.replace(k, v)

    return sentence


# def emotions_to_ids(emo):
#     emo_ids = []
#     for e in emo:
#         if e == "joyful":
#             emo_ids.append("0")
#         elif e == "excited":
#             emo_ids.append("1")
#         elif e == "proud":
#             emo_ids.append("2")
#         elif e == "grateful":
#             emo_ids.append("3")
#         elif e == "hopeful":
#             emo_ids.append("4")
#         elif e == "surprised":
#             emo_ids.append("5")
#         elif e == "confident":
#             emo_ids.append("7")
#         elif e == "content":
#             emo_ids.append("6")
#         elif e == "impressed":
#             emo_ids.append("8")
#         elif e =="trusting":
#             emo_ids.append("9")
#         elif e =="faithful":
#             emo_ids.append("10")
#         elif e == "prepared":
#             emo_ids.append("11")
#         elif e == "caring":
#             emo_ids.append("12")
#         elif e == "anticipating":
#             emo_ids.append("14")
#         elif e == "sentimental":
#             emo_ids.append("15")
#         elif e == "anxious":
#             emo_ids.append("16")
#         elif e == 'apprehensive':
#             emo_ids.append("17")
#         elif e == "nostalgic":
#             emo_ids.append("18")
#         elif e ==  "devastated":
#             emo_ids.append("13")
#         elif e == "lonely":
#             emo_ids.append("19")
#         elif e == "embarrassed":
#             emo_ids.append("20")
#         elif e == "ashamed":
#             emo_ids.append("21")
#         elif e =="guilty":
#             emo_ids.append("22")
#         elif e == "sad":
#             emo_ids.append("23")
#         elif e == "jealous":
#             emo_ids.append("24")
#         elif e == "terrified":
#             emo_ids.append("25")
#         elif e == "afraid":
#             emo_ids.append("26")
#         elif e == "disappointed":
#             emo_ids.append("27")
#         elif e == "angry":
#             emo_ids.append("28")
#         elif e == "annoyed":
#             emo_ids.append("29")
#         elif e ==  "disgusted":
#             emo_ids.append("30")
#         elif e == "furious":
#             emo_ids.append("31")
#         else:
#             print(e)
#     return emo_ids
#
# def emotions_to_ids(emo):
#     emo_ids = []
#     for e in emo:
#         e = str(e)
#         if e in ["joyful","excited","proud","grateful","hopeful","surprised","content"]:
#             emo_ids.append("0")
#         elif e in ["confident","impressed","trusting","faithful","prepared","caring","anticipating"]:
#             emo_ids.append("0")
#         elif e in["sentimental", "anxious",'apprehensive',"nostalgic","devastated"]:
#             emo_ids.append("1")
#         elif e in ["lonely","embarrassed","ashamed","guilty","sad","jealous","terrified","afraid"]:
#             emo_ids.append("1")
#         elif e in ["disappointed","angry","annoyed","disgusted","furious"]:
#             emo_ids.append("1")
#         else:
#             print(e)
#     return emo_ids

def load_conversations(data,type):
    action_dict = {}
    for i, act in enumerate(DIALOG_ACTIONS):
        action_dict[act] = i
    emo_dict = {}
    for i, emo in enumerate(EMO_LABELS):
        emo_dict[emo] = i
    conversations = []
    conversations_length=[]
    conversations_turns=[]
    #sentences_length=[]
    Speakers=[]
    ids=[]
    emotions=[]
    actions=[]
    #conv_id,utterance_idx,context,prompt,speaker_idx,utterance,selfeval,tags
    for i,dialog in enumerate(data):
        # if i>2000:
        #     break
        if dialog["utterances"] == []:
            continue
        id=[]
        conv = []
        emo = []
        act = []
        speaker=[]
        for utterances in dialog["utterances"]:
            one_conv=[]
            id.append(str(type)+"_"+utterances["id"])
            one_conv.append(utterances["history"][-1].lower().strip())
            one_conv.append(utterances["reply"][0].lower().strip())
            emo.append(emo_dict[utterances["speaker_emotion"][0]])
            emo.append(emo_dict[utterances["listener_emotion"][0]])
            act.append(action_dict[utterances["speaker_action"][0]])
            act.append(action_dict[utterances["listener_action"][0]])
            conv.append(one_conv)
            speaker.append([[0,1],[1,0]])
        ids.append(id)
        conversations.append(conv)
        emotions.append(emo)
        actions.append(act)
        conversations_length.append((len(emo)))
        conversations_turns.append(len(emo) // 2)
        Speakers.append(speaker)
    #加一个长度关系的判断
    return ids,conversations,conversations_length,conversations_turns,Speakers,emotions,actions



def load_emotion(fileName,spliter=" "):
    emotion_labels=[]
    with open(fileName, 'r') as f:
        for line in f:
            emotion_class=[]
            line = line.strip()
            if not line:
                continue
            emotion = line.split(spliter)
            if len(emotion) %2 !=0 :
               emotion=emotion[:-1]
            for label in emotion:
                zero=np.zeros(shape=7)
                zero[int(label)]=1
                emotion_class.append(zero)
            emotion_=np.reshape(emotion_class,(-1,2,7))
            emotion_labels.append(emotion_)
    return emotion_labels


def tokenize_conversation(lines):
    sentence_list=[]
    for line in lines:
        one_s = []
        for one in line:
            s=tokenizer(one)
            one_s.append(s)
        sentence_list .append(one_s)
    return sentence_list

def pad_sentences(conversations, max_sentence_length=40):
    def pad_tokens(one_turn, max_sentence_length=max_sentence_length):
        one_turn_tokens=[]
        for tokens in one_turn:
            n_valid_tokens = len(tokens)
            if n_valid_tokens > max_sentence_length - 1:
                tokens = tokens[:max_sentence_length - 1]
            n_pad = max_sentence_length - n_valid_tokens - 1
            tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
            one_turn_tokens.append(tokens)
        return one_turn_tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence)  for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        sentence_length =np.reshape( [min(len(sentence)+1 , max_sentence_length) # +1 for EOS token
                            for one in conversation for sentence in one],(-1,2))
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument('-s', '--max_sentence_length', type=int, default=40)
    #parser.add_argument('-c', '--max_conversation_length', type=int, default=10)

    # Vocabulary
    parser.add_argument('--max_vocab_size', type=int, default=30000)
    parser.add_argument('--min_vocab_frequency', type=int, default=3)
    parser.add_argument('--n_workers', type=int, default=6)
    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    #max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers

    print("Loading conversations...")
    all_data=json.load(open(datasets_dir.joinpath("ED_alter_labels.json"),encoding="utf-8"))

    train_ids,train, train_length,train_turns,train_Speakers,train_emotions,train_actions= load_conversations(all_data["train"],"tr")
    valid_ids, valid, valid_length, valid_turns,valid_Speakers, valid_emotions,valid_actions = load_conversations(all_data["valid"], "va")
    test_ids, test, test_length, test_turns,test_Speakers, test_emotions,test_actions = load_conversations(all_data["test"],"te")

    #all_ids=train_ids+valid_ids+test_ids
    #all_emotion_labels=train_emotions+valid_emotions+test_ids
    #all_conv_length=train_length+valid_length+test_length
    #all_speakers=train_Speakers+valid_Speakers+test_Speakers
    #all_conversations=train+valid+test

    print("#train:%s, #val:%s, #test:%s"%(np.shape(train), np.shape(valid), np.shape(test)))
    def to_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    with Pool(n_workers) as pool:
        train_conversations_tokens = list(tqdm(pool.imap(tokenize_conversation, train),
                                  total=len(train)))
        valid_conversations_tokens = list(tqdm(pool.imap(tokenize_conversation, valid),
                                  total=len(valid)))
        test_conversations_tokens = list(tqdm(pool.imap(tokenize_conversation, test),
                                  total=len(test)))
    pool.close()

    train_sentences, train_sentence_length = pad_sentences(train_conversations_tokens,
                                                          max_sentence_length=max_sent_len)
    valid_sentences, valid_sentence_length = pad_sentences(valid_conversations_tokens,
                                                           max_sentence_length=max_sent_len)
    test_sentences, test_sentence_length = pad_sentences(test_conversations_tokens,
                                                           max_sentence_length=max_sent_len)
    train_data=[train_sentences,train_sentence_length,train_length,train_turns,train_Speakers,train_emotions,train_actions]
    valid_data = [valid_sentences, valid_sentence_length, valid_length, valid_turns,valid_Speakers, valid_emotions,valid_actions]
    test_data = [test_sentences, test_sentence_length, test_length, test_turns,test_Speakers, test_emotions,test_actions]
    for split_type, conv_objects in [('train', train_data), ('validation', valid_data), ('test', test_data)]:
        print(f'Processing {split_type} dataset...')
        split_data_dir = datasets_dir.joinpath(split_type)
        #to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))
        to_pickle(conv_objects, split_data_dir.joinpath('data.pkl'))

    print('Save Vocabulary...')
    vocab = Vocab(tokenizer)
    vocab.add_dataframe(train_sentences+valid_sentences)
    vocab.update(max_size=max_vocab_size, min_freq=min_freq)

    print('Vocabulary size: ', len(vocab))
    vocab.pickle(datasets_dir.joinpath('word2id.pkl'), datasets_dir.joinpath('id2word.pkl'))

    print('Done!')

