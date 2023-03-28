import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
from layers.rnncells import StackedLSTMCell, StackedGRUCell

project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
data_dir = project_dir.joinpath('datasets')
data_dict = {'ijcnlp_dailydialog': data_dir.joinpath('ijcnlp_dailydialog'), 'ED': data_dir.joinpath('ED')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
rnn_dict = {'lstm': nn.LSTM, 'gru': nn.GRU}
rnncell_dict = {'lstm': StackedLSTMCell, 'gru': StackedGRUCell}
username = Path.home().name
save_dir = project_dir.joinpath('datasets')

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'rnn':
                    value = rnn_dict[value]
                if key == 'rnncell':
                    value = rnncell_dict[value]
                setattr(self, key, value)

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data]

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir.joinpath(self.mode)
        # Pickled Vocabulary
        self.word2id_path = self.dataset_dir.joinpath('word2id.pkl')
        self.id2word_path = self.dataset_dir.joinpath('id2word.pkl')
        self.glove_path = project_dir.joinpath('datasets/glove.840B.300d.txt')
        # self.checkpoint=self.dataset_dir.joinpath('ETHREED/model/27.77.pkl')
        #7 Pickled Dataframe
        self.sentences_path = self.data_dir.joinpath('data.pkl')
        self.sentences_path_100 = self.data_dir.joinpath('data_sample.pkl')

        # Save path
        if self.mode == 'train' and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = project_dir.joinpath(self.data, self.model)
            self.save_path=save_dir.joinpath(self.save_path,"model")
            self.save_emo_path=save_dir.joinpath(self.data, "emotion")
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()
    #fine_tune
    parser.add_argument('--test', type=str2bool, default=False)
    parser.add_argument('--R_classification', type=str2bool, default=True)
    parser.add_argument('--listener_state', type=str2bool, default=False)
    parser.add_argument('--emotion_policy', type=str2bool, default=True)
    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_classes', type=int, default=32)
    parser.add_argument('--behavior_n_classes', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--emo_checkpoint', type=str, default=None)
    parser.add_argument('--emo_classifier', type=str2bool, default=True)

    # Generation
    parser.add_argument('--max_unroll', type=int, default=40)
    parser.add_argument('--sample', type=str2bool, default=False,
                        help='if false, use beam search for decoding')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--out_drop', type=float, default=0.5)
    #pointer
    parser.add_argument('--enc_attn', type=str2bool, default=True)
    parser.add_argument('--dec_attn', type=str2bool, default=False)
    parser.add_argument('--pointer', type=str2bool, default=True)
    parser.add_argument('--enc_total_size', type=int, default=600)

    #cover_func: str = 'max'  # how to aggregate previous attention distributions? sum or max
    # cover_loss: float = 1  # add coverage loss if > 0; weight of coverage loss as compared to NLLLoss
    # show_cover_loss: bool = False  # include coverage loss in the loss shown in the progress bar?
    # coverage
    parser.add_argument('--enc_attn_cover', type=str2bool, default=True)
    parser.add_argument('--cover_loss', type=float, default=1)
    parser.add_argument('--cover_func', type=str, default="max")
    parser.add_argument('--show_cover_loss', type=str2bool, default=True)



    # Model
    parser.add_argument('--model', type=str, default='ETHREED')
    # Currently does not support lstm
    parser.add_argument('--rnn', type=str, default='gru')
    parser.add_argument('--rnncell', type=str, default='gru')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--tie_embedding', type=str2bool, default=True)
    parser.add_argument('--encoder_hidden_size', type=int, default=300)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--decoder_hidden_size', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--context_size', type=int, default=300)#与内容输出的维度相同
    parser.add_argument('--feedforward', type=str, default='FeedForward')
    parser.add_argument('--activation', type=str, default='Tanh')
    #self-attention
    parser.add_argument('--self_attention_hidden_size', type=int, default=600)
    parser.add_argument('--self_attention_dropout', type=float, default=0.5)
    parser.add_argument('--self_attention_head', type=int, default=2)

    # VAE model
    parser.add_argument('--z_sent_size', type=int, default=300)
    parser.add_argument('--z_conv_size', type=int, default=300)
    parser.add_argument('--word_drop', type=float, default=0.0,
                        help='only applied to variational models')
    parser.add_argument('--kl_threshold', type=float, default=0.0)
    parser.add_argument('--kl_annealing_iter', type=int, default=25000)
    parser.add_argument('--importance_sample', type=int, default=300)
    parser.add_argument('--sentence_drop', type=float, default=0.5)
    #Feature Extraction
    parser.add_argument('--D_m', type=int, default=600)
    parser.add_argument('--D_g', type=int, default=300)
    parser.add_argument('--D_p', type=int, default=300)
    parser.add_argument('--D_r', type=int, default=300)
    parser.add_argument('--D_e', type=int, default=300)
    parser.add_argument('--D_b', type=int, default=300)
    parser.add_argument('--D_h', type=int, default=300)
    parser.add_argument('--D_a', type=int, default=300)
    parser.add_argument('--context_attention', type=str, default='simple')
    parser.add_argument('--dropout_rec', type=float, default='0.5')

    # Generation
    parser.add_argument('--n_context', type=int, default=1)
    parser.add_argument('--n_sample_step', type=int, default=1)

    # BOW
    parser.add_argument('--bow', type=str2bool, default=False)

    # Utility
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)

    # Data
    parser.add_argument('--data', type=str, default='ED')


    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
