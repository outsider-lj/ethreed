from collections import defaultdict
import pickle
import torch
from torch import Tensor
from torch.autograd import Variable
from nltk import FreqDist
from .convert import to_tensor, to_var
import numpy as np

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]

class OOVDict(object):

  def __init__(self, base_oov_idx):
    self.word2id = {}  # type: Dict[Tuple[int, str], int]
    self.id2word = {}  # type: Dict[Tuple[int, int], str]
    self.next_index = {}  # type: Dict[int, int]
    self.base_oov_idx = base_oov_idx
    self.ext_vocab_size = base_oov_idx

  def add_word(self, idx_in_batch, word) -> int:
    key = (idx_in_batch, word)
    index = self.word2id.get(key)
    if index is not None: return index
    index = self.next_index.get(idx_in_batch, self.base_oov_idx)
    self.next_index[idx_in_batch] = index + 1
    self.word2id[key] = index
    self.id2word[(idx_in_batch, index)] = word
    self.ext_vocab_size = max(self.ext_vocab_size, index + 1)
    return index

class Vocab(object):
    def __init__(self, tokenizer=None, max_size=None, min_freq=1):
        """Basic Vocabulary object"""

        self.vocab_size = 0
        self.freqdist = FreqDist()
        self.tokenizer = tokenizer
        self.embeddings = None

    def update(self, max_size=None, min_freq=1):
        """
        Initialize id2word & word2id based on self.freqdist
        max_size include 4 special tokens
        """

        # {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.id2word = {
            PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN,
            SOS_ID: SOS_TOKEN, EOS_ID: EOS_TOKEN
        }
        # {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.word2id = defaultdict(lambda: UNK_ID)  # Not in vocab => return UNK
        self.word2id.update({
            PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
            SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        })
        # self.word2id = {
        #     PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID,
        #     SOS_TOKEN: SOS_ID, EOS_TOKEN: EOS_ID
        # }

        vocab_size = 4
        min_freq = max(min_freq, 2)

        # Reset frequencies of special tokens
        # [...('<eos>', 0), ('<pad>', 0), ('<sos>', 0), ('<unk>', 0)]
        freqdist = self.freqdist.copy()
        special_freqdist = {token: freqdist[token]
                            for token in [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]}
        freqdist.subtract(special_freqdist)

        # Sort: by frequency, then alphabetically
        # Ex) freqdist = { 'a': 4,   'b': 5,   'c': 3 }
        #  =>   sorted = [('b', 5), ('a', 4), ('c', 3)]
        sorted_frequency_counter = sorted(freqdist.items(), key=lambda k_v: k_v[0])
        sorted_frequency_counter.sort(key=lambda k_v: k_v[1], reverse=True)

        for word, freq in sorted_frequency_counter:

            if freq < min_freq or vocab_size == max_size:
                break
            self.id2word[vocab_size] = word
            self.word2id[word] = vocab_size
            vocab_size += 1

        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, item):
        if type(item) is int:
            return self.id2word.get(item)
        return self.word2id.get(item, UNK_ID)

    def load(self, word2id_path=None, id2word_path=None):
        if word2id_path:
            with open(word2id_path, 'rb') as f:
                word2id = pickle.load(f)
            # Can't pickle lambda function
            self.word2id = defaultdict(lambda: UNK_ID)
            self.word2id.update(word2id)
            self.vocab_size = len(self.word2id)

        if id2word_path:
            with open(id2word_path, 'rb') as f:
                id2word = pickle.load(f)
            self.id2word = id2word

    def add_word(self, word):
        assert isinstance(word, str), 'Input should be str'
        self.freqdist.update([word])

    def add_sentence(self, sentence, tokenized=False):
        if not tokenized:
            sentence = self.tokenizer(sentence)
        for word in sentence:
            self.add_word(word)

    def add_dataframe(self, conversation_df, tokenized=True):
        for conversation in conversation_df:
            for one in conversation:
                for sentence in one:
                    self.add_sentence(sentence, tokenized=tokenized)

    def pickle(self, word2id_path, id2word_path):
        with open(word2id_path, 'wb') as f:
            pickle.dump(dict(self.word2id), f)

        with open(id2word_path, 'wb') as f:
            pickle.dump(self.id2word, f)

    def to_list(self, list_like):
        """Convert list-like containers to list"""
        if isinstance(list_like, list):
            return list_like

        if isinstance(list_like, Variable):
            return list(to_tensor(list_like).numpy())
        elif isinstance(list_like, Tensor):
            return list(list_like.numpy())

    def id2sent(self,i, id_list,oov_dict):
        """list of id => list of tokens (Single sentence)"""
        id_list = self.to_list(id_list)
        sentence = []
        for id in id_list:
            if id == EOS_ID:
                break
            if id>= self.vocab_size:
                word=oov_dict.id2word.get((int(i/2),id),UNK_ID)
            else:
                word = self.id2word[id]
            if word not in [EOS_ID, SOS_ID, PAD_ID]:
                sentence.append(str(word))

        return sentence

    def sent2id(self, sentence, oov_dict,i,target=False,var=False):
        """list of tokens => list of id (Single sentence)"""
        id_list=[]
        for word in sentence:
            idx=self.word2id[word]
            if idx==UNK_ID:
                #self.add_word(word)
                if target==False:
                    if i%2 ==1:
                        idx = oov_dict.add_word(int(i/2), word)
                else:
                    idx=oov_dict.word2id.get((i, word), idx)
                    #if idx != 1:
                    #    print(idx)
            id_list.append(idx)
        if var:
            id_list = to_var(torch.LongTensor(id_list), eval=True)
        return id_list

    def decode(self,i, id_list,oov_dict):
        sentence = self.id2sent(i,id_list,oov_dict)#一个batch的第i个句子
        return ' '.join(sentence)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        #self.word2id={'<pad>':0,'<unk>':1,'<sos>':2,'eos':3}
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for (i,line) in enumerate(f):
                line = line.split()
                word = line[0].decode('utf-8')
                #self.word2id[word] =i+4
                idx = self.word2id.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[PAD_ID] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                num_embeddings += 1
        #self.id2word = {value: key for key, value in self.word2id.items()}

        return num_embeddings
