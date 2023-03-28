import json
import torch
from itertools import chain
from collections import defaultdict
from transformers import BertTokenizer,BertModel,BertConfig,RobertaTokenizer,RobertaConfig,RobertaModel
from pytorch_pretrained_bert import GPT2LMHeadModel,GPT2Tokenizer,GPT2Config
import torch.nn as nn
import argparse
SPECIAL_TOKENS = ["<speaker1>", "<speaker2>"]
DIALOG_ACTIONS=["acknowleding","agreeing","consoling","encouraging","questioning","suggesting","sympathizing","wishing","neutral"]
EMO_LABELS=["joyful","excited","proud", "grateful","hopeful","surprised","confident","content","impressed","trusting","faithful","prepared",
 "caring","devastated","anticipating","sentimental","anxious","apprehensive","nostalgic", "lonely","embarrassed", "ashamed","guilty",
 "sad","jealous", "terrified","afraid","disappointed","angry","annoyed","disgusted","furious"]
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def build_input_from_segments(history, reply, tokenizer, SPECIAL_TOKENS,  with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])

    instance = {}
    sequence = [[bos] + history[0]] + history[1:] + [
        reply + ([eos] if with_eos else [])]  # seq = [personas, history, reply] concatenate all persona sentences
    sequence = [sequence[0]] + [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                                enumerate(sequence[1:])]

    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in
                                  s]  # the last for is for repeating the speaker1 and speaker2 for all tokens
    return instance, sequence
class Roberta_emo(nn.Module):
    def __init__(self, config):
        super(Roberta_emo, self).__init__()
        roberta_config=RobertaConfig.from_pretrained(config.emo_model_checkpoint)
        self.feature=RobertaModel(roberta_config)
        emo_att = torch.Tensor(1, roberta_config.hidden_size)
        # topic_att = torch.Tensor(1, config.hidden_size)
        emo_att = torch.nn.init.uniform_(emo_att)
        # topic_att = torch.nn.init.uniform_(topic_att)
        self.emo_attention_vector = nn.Parameter(emo_att)
        self.dense = nn.Linear(roberta_config.hidden_size, roberta_config.hidden_size)
        self.dropout = nn.Dropout(roberta_config.hidden_dropout_prob)
        self.out_proj = nn.Linear(roberta_config.hidden_size, 32)
    def forward(self,input_ids=None,attention_mask=None,return_dict=True):
        output=self.feature(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        features=output.last_hidden_state
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
# def if_qustioning(sent):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="../datasets/ED/ED.json",
        type=str,
        # required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--emo_model_checkpoint",
        default="./roberta-large/emo_classifior",
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--save_dir",
        default="../datasets/ED/ED_add_labels.json",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )


    args = parser.parse_args()
    original_ed=json.load(open(args.data_dir,encoding="utf-8"))
    json_file = open(args.save_dir, 'w+', encoding='utf-8')
    new_data= {"train": [], "valid": [], "test": []}
    model_act_classifior=Roberta_action(args)
    model_act_classifior.to("cuda")
    tokenizer=RobertaTokenizer.from_pretrained(args.emo_model_checkpoint)
    model_act_classifior.load_state_dict(
        torch.load(
            args.act_model_checkpoint+"/pytorch_model.bin", map_location="cuda"
        ),
        strict=True,
    )
    model_emo_classifior = Roberta_emo(args)
    model_emo_classifior.to("cuda")
    model_emo_classifior.load_state_dict(
        torch.load(
            args.emo_model_checkpoint + "/pytorch_model.bin", map_location="cuda"
        ),
        strict=True,
    )
    action_dict = {}
    for i, act in enumerate(DIALOG_ACTIONS):
        action_dict[i] = act
    emo_dict = {}
    for i, emo in enumerate(EMO_LABELS):
        emo_dict[i] = emo
    print("processing data")
    for dataset_name, dataset in original_ed.items():
        for data in dataset:
            d= {"personality": [],"utterances":[]}
            emo_list=[]
            for dialog in data["utterances"]:
                instance={}
                # post=dialog[history][-1]
                post=dialog["history"][-1]
                reply=dialog["reply"]
                reply_input_ids=[tokenizer.encode(reply[0])]
                post_input_ids = [tokenizer.encode(reply[0])]
                r_emo_logits = model_emo_classifior(input_ids=torch.LongTensor(reply_input_ids).cuda())
                reply_emo_label = torch.argmax(r_emo_logits, dim=1).item()
                instance["id"]=dialog["id"]
                instance["situation"]=dialog["situation"]
                instance["history"]=dialog["history"]
                instance["speaker_emotion"]=dialog["speaker_emotion"]
                instance["listener_emotion"]=[emo_dict[reply_emo_label]]
                instance["reply"]=dialog["reply"]
                d["utterances"].append(instance)
            new_data[dataset_name].append(d)
    json_file.write(json.dumps(new_data, default=set_default, ensure_ascii=False))
    json_file.close()
    print("sucess")