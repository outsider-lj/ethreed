# Preprocess cornell movie dialogs dataset
import argparse
from pathlib import Path
import pandas as pd
project_dir = Path(__file__).resolve().parent
print(project_dir)
datasets_dir = project_dir.joinpath('ED/')
import csv
import json
from collections import defaultdict
import re
from itertools import chain
def get_one_instance(num,situation,utterance,emotion,history,one,more):
    instance = {}
    instance["history"] = history  # 用户字嵌入
    instance["speaker_emotion"] = [emotion]
    instance["situation"] = [situation]
    instance["reply"]=[utterance]
    instance["id"]=num
    #加一个长度关系的判断
    return instance,one,more
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
                   "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are","haven't":"have not"}
    for k, v in word_pairs1.items():
        sentence = sentence.replace(k, v)
    for k, v in word_pairs2.items():
        sentence = sentence.replace(k, v)

    return sentence

if __name__ == '__main__':
    json_file = open('../datasets/ED/ED.json', 'w+', encoding='utf-8')
    datasets = {"train":defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    for type in ["train","valid","test"]:
        k=0
        j=0
        one=0
        more=0
        fileName="../datasets/ED/"+type+".csv"
        dataframe = pd.read_csv(open(fileName), encoding="utf-8", delimiter="\t")
        for i in range(dataframe.size):
            line = dataframe.loc[i].values[0].split(",")
            id=line[0]
            num=line[1]
            emotion = line[2]
            situation = line[3].replace("_comma_", ",")
            utterance =line[5].replace("_comma_", ",")
            situation=clean(situation)
            utterance=clean(utterance)
            if int(num) == 1:
                if i==0:
                    conv={"personality":[], "utterances": []}
                else:
                    if k==0:
                        datasets[type]=[conv]
                        j = 0
                        k += 1
                    else:
                        datasets[type].append(conv)
                        j=0
                    conv={"personality": [], "utterances": []}
                emotions=[emotion]
                history = [utterance]
            elif int(num) % 2==1:
                history.append(utterance)
                emotions.append(emotion)
            else:
                instance={}
                e=emotions[:len(emotions)]
                h=history[:len(history)]
                instance,one,more=get_one_instance(id,situation,utterance,emotion,h,one,more)
                history.append(utterance)
                emotions.append(emotion)
                if j==0:
                    conv["utterances"]=[instance]
                    j+=1
                else:
                    conv["utterances"].append(instance)
                if i==dataframe.size-1:
                    datasets[type].append(conv)

    json_file.write(json.dumps(datasets, ensure_ascii=False))
    json_file.close()