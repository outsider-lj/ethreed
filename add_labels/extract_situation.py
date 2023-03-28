import json
ed_kg=json.load(open("../datasets/ED/ED.json",encoding="utf-8"))
num_emo_sent=0
num_topic_sent=0
num_all=0
d={}
with open("../datasets/ED/situation_train.txt", 'w') as f:
    for dialog in ed_kg["train"]:
        if dialog["utterances"]==[]:
            continue
        situation=dialog["utterances"][0]["situation"]
        emotion=dialog["utterances"][0]["speaker_emotion"]
        f.write(emotion[0]+"\t")
        f.write(situation[0]+"\n")