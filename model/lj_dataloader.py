from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
class DialogDataset(Dataset):#继承了Dataset类
    def __init__(self, all_data, data=None):
        # [total_data_size, max_conversation_length, max_sentence_length]
        # tokenized raw text of sentences
        self.sentences = all_data[0]
        #self.vocab = vocab
        #self.oov_dict=oov_dict

        # list of length of sentences
        # [total_data_size, max_conversation_length]
        # conversation length of each batch
        # [total_data_size]
        self.sentence_length = all_data[1]
        self.conversation_length = all_data[2]
        self.conversation_turns=all_data[3]
        self.speakers=all_data[4]
        self.emotion_labels=all_data[5]

        self.data = data
        self.len = len(self.sentences)

    def __getitem__(self, index):
        """Return Single data sentence"""
        # [max_conversation_length, max_sentence_length]
        sentence = self.sentences[index]
        sentence_length = self.sentence_length[index]
        conversation_length = self.conversation_length[index]
        conversation_turns = self.conversation_turns[index]
        speakers=self.speakers[index]
        emotion_labels=self.emotion_labels[index]
        # word => word_ids
        #sentence = self.sent2id(sentence,self.oov_dict)

        return sentence, sentence_length,conversation_length,conversation_turns,speakers,emotion_labels

    def __len__(self):
        return self.len

    def sent2id(self, sentences,oov_dict):
        """word => word id"""
        # [max_conversation_length, max_sentence_length]
        sentences_tokenid=[]
        for one_turn in sentences:
            one_turn_tokenid=[]
            for sentence in one_turn:
                one_turn_tokenid.append(self.vocab.sent2id(sentence,oov_dict))
            sentences_tokenid.append(one_turn_tokenid)
        return sentences_tokenid

def pad_tensor(vec, pad, dim):
    if isinstance(vec,int):
        return  vec
    else:
        vec=np.array(vec)
        pad_size = list(vec.shape)
        pad_size[dim] = pad - pad_size[dim]
    return np.concatenate([vec, np.zeros(pad_size)], axis=dim)
def get_loader(all_data,  batch_size=80, data=None, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    def collate_fn(data):
        """
        Collate list of data in to batch

        Args:
            data: list of tuple(source, target, conversation_length, source_length, target_length)
        Return:
            Batch of each feature
            - source (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - target (LongTensor): [batch_size, max_conversation_length, max_source_length]
            - conversation_length (np.array): [batch_size]
            - source_length (LongTensor): [batch_size, max_conversation_length]
        """
        # Sort by conversation length (descending order) to use 'pack_padded_sequence'

        data.sort(key=lambda x: x[2], reverse=True)
        # Separate
        sentences, sentence_length, conversation_length,conversation_turns,speakers,emotion_labels= zip(*data)
        #print(sentences)
        #sentences=np.array(sentences).transpose(1, 0, 2, 3)
        #input_sentences=sentences[:-1]
        #target_sentences=sentences[1:,:,1:,:]
        #sentence_length=np.array(sentence_length).transpose(1,0,2)[:-1]
        #speakers=np.array(speakers).transpose(1,0,2,3)[:-1]
        # return sentences, conversation_length, sentence_length.tolist()
        return sentences, sentence_length, conversation_length,conversation_turns,speakers, emotion_labels

    dataset = DialogDataset(all_data,  data=data)#实际只运行了init中的方法，进行赋值
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True)
    return data_loader
