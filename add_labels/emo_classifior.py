# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import logging
import math
import os
from collections import defaultdict
from itertools import chain
from pprint import pformat

import numpy as np
import json
import torch
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear,create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
import argparse
from transformers import RobertaTokenizer,RobertaConfig,RobertaForSequenceClassification,AdamW,RobertaModel,AutoTokenizer
import torch.nn as nn

# BOS_TOKEN_ID=0
PAD_TOKEN_ID=1
# EOS_TOKEN_ID=2
MAX_LENGTH = 40
n = ['NN','NNP','NNPS','NNS','UH']#5
v = ['VB','VBD','VBG','VBN','VBP','VBZ']#6
a = ['JJ','JJR','JJS']#3
r = ['RB','RBR','RBS','RP','WRB']#5
EMO_LABELS =["impressed","grateful","prepared","excited","disappointed","afraid","disgusted","annoyed"]
MODEL_INPUTS = ["situation","attention_mask","emotion"]  # , "token_type_ids","mc_token_ids", "mc_labels"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
EMO_LABELS=["joyful","excited","proud", "grateful","hopeful","surprised","confident","content","impressed","trusting","faithful","prepared",
 "caring","devastated","anticipating","sentimental","anxious","apprehensive","nostalgic", "lonely","embarrassed", "ashamed","guilty",
 "sad","jealous", "terrified","afraid","disappointed","angry","annoyed","disgusted","furious"]
emo2id={}
for i,emo in enumerate(EMO_LABELS):
    emo2id[emo]=i
logger = logging.getLogger(__file__)
def get_losses_weights(losses:[list, np.ndarray, torch.Tensor]):
	if type(losses) != torch.Tensor:
		losses = torch.tensor(losses)
	weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
	return weights

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def average_distributed_scalar(scalar, config):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if config.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=config.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

def get_data_loaders(args, tokenizer):
    pad=tokenizer.convert_tokens_to_ids("<pad>")
    """ Prepare the dataset for training and evaluation """
    data = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    logger.info("Build inputs and labels")
    for type in ["train", "valid", "test"]:
        situaiton_path = args.data_dir+"/situation_" + type + ".txt"
        with open(situaiton_path, "r", encoding="utf-8") as f:
            for line in f:
                emotion, situation = line.split("\t")
                situation_tokens = tokenizer.encode(situation)
                if len(situation_tokens)>MAX_LENGTH:
                    situation_tokens=situation_tokens[:MAX_LENGTH]
                else:
                    situation_tokens=situation_tokens+(MAX_LENGTH-len(situation_tokens))*[pad]
                emotion_id = emo2id[emotion]
                data[type]["emotion"].append(emotion_id)
                data[type]["situation"].append(situation_tokens)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": [], "test": []}
    for dataset_name, dataset in data.items():
        # dataset = pad_dataset(dataset, padding=pad)
        for input_name in MODEL_INPUTS:
            if input_name == "attention_mask":
                for inputs in dataset["situation"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["attention_mask"].append(att_mask)
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["test"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else torch.utils.data.RandomSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else torch.utils.data.RandomSampler(valid_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler
class Roberta_emo(nn.Module):
    def __init__(self, config):
        super(Roberta_emo, self).__init__()
        roberta_config=RobertaConfig.from_pretrained(config.model_checkpoint)
        self.feature=RobertaModel.from_pretrained(config.model_checkpoint)
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
        # encoder_states=output.last_hidden_state
        # inverted_mask = 1.0 - attention_mask  # .unsqueeze(1)
        # inverted_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(encoder_states.dtype).min)
        # emo_att = torch.matmul(self.emo_attention_vector,
        #                        encoder_states.transpose(1, 2)) + inverted_mask.unsqueeze(1)
        # emotion_focused_attention = nn.functional.softmax(emo_att,
        #                                                   -1)  # torch.mul(attention_mask.unsqueeze(1), emo_att)
        # emo_att = self.dropout(emotion_focused_attention)
        # x = torch.matmul(emo_att, encoder_states)
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
def train():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="../datasets/ED",
        type=str,
        # required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_checkpoint",
        default="./roberta-large",
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--log_dir",
        default="./roberta-large/emo_classifior",
        type=str,
        # required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=40,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_predict", default=True, help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        default=True,
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", default=True, help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents.",
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents.",
    )
    # parser.add_argument(
    #     "--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.",
    # )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--valid_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--optimizer", default="AdamW", type=str, help="Optimizer (AdamW or lamb)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr", default=5e-6, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--n_epochs", default=5, type=int, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--max_norm", default=1.0, type=float)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_before_start", default=False, help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    args.distributed = (args.local_rank != -1)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    tokenizer =AutoTokenizer.from_pretrained(args.model_checkpoint,use_fast=True)
    model=Roberta_emo(args)
    model.to(args.device)
    # a=model.parameters()
    optimizer = AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to("cuda") for input_tensor in batch)
        input_ids, attention_mask,emo_label= batch
        emo_logits= model(input_ids=input_ids,
                            # encoder_mask_matrix=mask_matrix,
                            attention_mask=attention_mask,
                       return_dict=True)  # 进入model的forward
        emo_logits_flat_shifted = emo_logits.contiguous().view(-1, emo_logits.size(-1))
        emo_labels_flat_shifted = emo_label.contiguous().view(-1)
        loss = torch.nn.CrossEntropyLoss()(emo_logits_flat_shifted,emo_labels_flat_shifted)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        # f = open(os.path.join('bart-base/test_results.txt'), 'a')
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, attention_mask,  emo_label = batch
            emo_logits = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=True )  # 进入model的forward
            emo_logits_flat_shifted = emo_logits.contiguous().view(-1, emo_logits.size(-1))
            emo_labels_flat_shifted = emo_label.contiguous().view(-1)
        return (emo_logits_flat_shifted,emo_labels_flat_shifted)
    trainer = Engine(update)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the  learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    # scheduler=create_lr_scheduler_with_warmup(optimizer,warmup_start_value=0,warmup_end_value=config.lr,warmup_duration=10)
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    # RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "loss_emo")
    metrics = {"nll1": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0], x[1])),
                "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll1": MetricsLambda(average_distributed_scalar, metrics["nll1"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"],args)})
    metrics["average_ppl1"] = MetricsLambda(math.exp, metrics["average_nll1"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=args.log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                                                              another_engine=trainer),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(args.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, args.log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).feature.config.to_json_file(os.path.join(args.log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(args.log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        # print(checkpoint_handler._saved[-1][1])
        os.rename(os.path.join(args.log_dir, checkpoint_handler._saved[-1][1]), os.path.join(args.log_dir,
                                                                                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()

