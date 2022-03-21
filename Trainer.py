import os
import re
import csv
from tqdm import tqdm
from torch.cuda.amp import autocast
import math
import numpy as np
from evaluations import eval_distinct, corpus_bleu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from models.PointerGeneratorTransformer import PointerGeneratorTransformer

import random
from utils import *
from preprocess import *
from models.model_utils import padding_trg


random.seed(2021)
torch.manual_seed(2021)
torch.cuda.manual_seed_all(2021)
np.random.seed(2021)

class Trainer(object):
    def __init__(self, args, rank=0):
        super(Trainer, self).__init__()
        self.dataset_dir = args.dataset_dir
        self.max_len = args.max_len
        self.tgt_len = args.tgt_len
        self.world_size = args.gpus
        # self.rank = rank
        self.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.epochs = args.epochs
        self.label_smooth = args.label_smooth

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.vocab = self.tokenizer.vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        self.pad_id = self.vocab['[PAD]']
        self.cls_id = self.vocab['[CLS]']
        self.sep_id = self.vocab['[SEP]']
        self.unk_id = self.vocab['[UNK]']

        self.train_data = self.load_data(args.train_file, args.train_file.split(".")[0]+".pt", is_test=False)
        self.dev_data = self.load_data(args.dev_file, args.dev_file.split(".")[0]+".pt", is_test=False)

        self.test_data = self.load_data(args.test_file, args.test_file.split(".")[0]+".pt")
        self.model = PointerGeneratorTransformer(
            rank=self.rank, src_vocab_size=self.vocab_size,
            tgt_vocab_size=self.vocab_size, inv_vocab=self.inv_vocab,
            pad_id=self.pad_id, max_len=self.max_len
        )
        self.fp16 = args.fp16

        # initialize model parameters
        self.init_parameters()

        self.logger = get_logger()

    def load_data(self, file_name, loader_name, is_test=False):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f'Loading data from {loader_file}')
            data = torch.load(loader_file)
        else:
            print(f'Construct data from {os.path.join(self.dataset_dir, file_name)}')
            src_data = []
            per_data = []
            if not is_test:
                trg_data = []

            with open(os.path.join(self.dataset_dir, file_name), 'r', encoding='utf-8') as f:
                r = csv.reader(f, delimiter='\t')

                for line in tqdm(r):
                    src = line[1].strip()
                    per = line[0].strip()
                    # add preprocess progress
                    # src = toSimpleChinese(src)
                    src_data.append(src)
                    per_data.append(per)
                    if not is_test:
                        trg = line[2].strip()
                        # trg = toSimpleChinese(trg)
                        trg_data.append(trg)

            query_encoded_dict = self.tokenizer.batch_encode_plus(src_data, add_special_tokens=True, max_length=self.max_len,
                                                            padding='max_length',
                                                            return_attention_mask=True, truncation=True,
                                                            return_tensors='pt')
            per_encoded_dict = self.tokenizer.batch_encode_plus(per_data, add_special_tokens=True, max_length=self.max_len,
                                                            padding='max_length',
                                                            return_attention_mask=True, truncation=True,
                                                            return_tensors='pt')
            query_input_ids = query_encoded_dict['input_ids']
            query_attention_masks = query_encoded_dict['attention_mask']
            query_type_ids = query_encoded_dict['token_type_ids'] * 0

            per_input_ids = per_encoded_dict['input_ids']
            per_attention_masks = per_encoded_dict['attention_mask']
            per_type_ids = per_encoded_dict['token_type_ids'] * 0 + 1

            src_input_ids = torch.cat([per_input_ids, query_input_ids], -1)
            src_attention_masks = torch.cat([per_attention_masks, query_attention_masks], -1)
            src_type_ids = torch.cat([per_type_ids, query_type_ids], -1)

            if not is_test:
                trg_input_ids, trg_ground_ids, trg_attention_masks = [], [], []
                for text in trg_data:
                    # encode text without trunction
                    encoded_text = self.tokenizer(text)
                    trg_ids, trg_attention_mask = encoded_text['input_ids'], encoded_text['attention_mask']
                    if len(trg_ids) > self.tgt_len:
                        trg_ids = trg_ids[:self.tgt_len - 1] + [self.sep_id]
                        trg_attention_mask = trg_attention_mask[:self.tgt_len]

                    # add padding
                    trg_input, trg_ground, trg_mask = padding_trg(trg_ids[:-1], trg_ids[1:], trg_attention_mask[:-1],
                                                                  self.tgt_len)

                    trg_input_ids.append(trg_input)
                    trg_ground_ids.append(trg_ground)
                    trg_attention_masks.append(trg_mask)

                data = {
                    'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks,
                    'trg_input_ids': torch.tensor(trg_input_ids), 'trg_ground_ids': torch.tensor(trg_ground_ids),
                    'trg_attention_masks': torch.tensor(trg_attention_masks), 'src_type_ids': src_type_ids,
                    'per_input_ids': per_input_ids, 'query_input_ids': query_input_ids
                }
            else:
                data = {
                    'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks
                }
        torch.save(data, loader_file)
        return data

    def load_test_data(self, file_name, loader_name):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f'Loading data from {loader_file}')
            data = torch.load(loader_file)
        else:
            print(f'Construct data from {os.path.join(self.dataset_dir, file_name)}')
            src_data = []
            per_data = []
            with open(os.path.join(self.dataset_dir, file_name), 'r', encoding='utf-8') as f:
                data = csv.reader(f, delimiter='\t')
                for line in tqdm(data):
                    # add preprocess progress
                    # src = toSimpleChinese(line.strip())
                    src = line[1].strip()
                    per = line[0].strip()
                    src_data.append(src + per)
                    per_data.append(per)

            encoded_dict = self.tokenizer.batch_encode_plus(src_data, add_special_tokens=True, max_length=self.max_len,
                                                            padding='max_length',
                                                            return_attention_mask=True, truncation=True,
                                                            return_tensors='pt')
            src_input_ids = encoded_dict['input_ids']
            src_attention_masks = encoded_dict['attention_mask']
            data = {
                'src_input_ids': src_input_ids, 'src_attention_masks': src_attention_masks
            }
            torch.save(data, loader_file)
        return data

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size, shuffle=True):
        if "trg_input_ids" in data_dict:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_attention_masks"],
                                    data_dict["trg_input_ids"], data_dict["trg_ground_ids"],
                                    data_dict["trg_attention_masks"], data_dict["src_type_ids"],
                                    data_dict["per_input_ids"], data_dict["query_input_ids"])
        else:
            dataset = TensorDataset(data_dict["src_input_ids"], data_dict["src_attention_masks"])
        if shuffle:
            # sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
            # dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataset_loader

    def cal_performance(self, logits, ground, smoothing=True):
        logits = logits.contiguous().view(-1, logits.size(-1))
        ground = ground.contiguous().view(-1)

        loss = self.cal_loss(logits, ground, smoothing=smoothing)

        pad_mask = ground.ne(self.pad_id)
        pred = logits.max(-1)[1]
        correct = pred.eq(ground)
        correct = correct.masked_select(pad_mask).sum().item()
        total_words = pad_mask.sum().item()
        return loss, correct, total_words

    def cal_loss(self, logits, ground, smoothing=True):
        def label_smoothing(logits, labels):
            eps = self.label_smooth
            num_classes = logits.size(-1)

            # >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
            # >>> z
            # tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
            #        [ 0.0000,  0.0000,  0.0000,  1.2300]])
            one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
            log_prb = F.log_softmax(logits, dim=1)
            non_pad_mask = ground.ne(self.pad_id)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()
            return loss

        if smoothing:
            loss = label_smoothing(logits, ground)
            # loss = F.cross_entropy(logits, ground, ignore_index=self.pad_id, label_smoothing=self.label_smooth)
        else:
            loss = F.cross_entropy(logits, ground, ignore_index=self.pad_id)

        return loss

    def init_parameters(self):
        for name, param in self.model.named_parameters():
            if 'encoder' not in name and 'tgt_embed' not in name and param.dim() > 1:
                xavier_uniform_(param)

    def train(self):
        train_loader = self.make_dataloader(0, self.train_data, self.train_batch_size)
        dev_loader = self.make_dataloader(0, self.dev_data, self.eval_batch_size)

        if os.path.exists('./model_dict/model.pt'):
            print("loading the checkpoint")
            model = torch.load('./model_dict/model.pt').to(self.rank)
        else:
            print("training a new model")
            model = self.model.to(self.rank)

        # model = self.model.to(self.rank)
        print(model)
        print(get_parameter_number(model))

        total_steps = len(train_loader) * self.epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': model.encoder.parameters(), 'lr': 7e-6, 'weight_decay': 0.01},
            {'params': model.tgt_embed.parameters(), 'lr': 7e-6, 'weight_decay': 0.01},
            {'params': model.decoder.parameters(), 'weight_decay': 0.01},
            {'params': model.p_vocab.parameters(), 'weight_decay': 0.01},
            {'params': model.p_gen.parameters(), 'weight_decay': 0.01}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        if self.fp16 == True:
            scaler = torch.cuda.amp.GradScaler()
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4000, num_training_steps=total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps / 10,
                                                    num_training_steps=total_steps)

        print(f"encoder lr:{optimizer.state_dict()['param_groups'][0]['initial_lr']}\n"
              f"tgt_embed lr:{optimizer.state_dict()['param_groups'][1]['initial_lr']}\n"
              f"other lr:{optimizer.state_dict()['param_groups'][2]['initial_lr']}\n"
              f"scheduler:linear warmup,warmup_steps={total_steps / 10}")

        is_best = False
        curr_valid_loss = 0
        best_valid_loss = float("inf")
        epochs_no_improve = 0

        total_steps = 0
        for epoch in range(self.epochs):
            print(f'Epoch / Total epochs: {epoch + 1} / {self.epochs}')
            running_loss = 0.0
            model.train()
            correct_words = 0
            total_words = 0
            for batch in tqdm(train_loader):
                src_input_ids, src_input_masks, src_type_ids = batch[0].to(self.rank), batch[1].to(self.rank), batch[
                    5].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[
                    4].to(self.rank)

                if self.fp16 == True:
                    # print("fp16")
                    with autocast():
                        outputs = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks, src_type_ids)

                        loss, n_correct, n_word = self.cal_performance(outputs, trg_ground_ids, smoothing=True)
                    # 对loss进行缩放，针对缩放后的loss进行反向传播
                    # （此部分计算在autocast()作用范围以外）
                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # 将梯度值缩放回原尺度后，优化器进行一步优化
                    scaler.step(optimizer)

                    # 更新scalar的缩放信息
                    scaler.update()
                else:
                    outputs = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks, src_type_ids)

                    loss, n_correct, n_word = self.cal_performance(outputs, trg_ground_ids, smoothing=True)

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()

                # print("lr[0]:",optimizer.state_dict()['param_groups'][0]["lr"])
                # print("initial_lr[0]:",optimizer.state_dict()['param_groups'][0]["initial_lr"])
                optimizer.zero_grad()
                scheduler.step()

                running_loss += loss.item()

                correct_words += n_correct
                total_words += n_word
                total_steps += 1
                if total_steps % 100 == 0:
                    self.logger.info(
                        f"Train Epoch: {epoch + 1}, Total Steps: {total_steps}, avg loss: {running_loss / total_steps:.4f}, accuracy: {100 * correct_words / total_words:.2f}%")

            # print statistics
            if total_steps % 100 == 0 or epoch % 1 == 0:
                self.logger.info(
                    f"Train Epoch: {epoch + 1}, avg loss: {running_loss / len(train_loader):.4f}, accuracy: {100 * correct_words / total_words:.2f}%")

            if epoch % 1 == 0:
                epochs_no_improve += 1
                curr_valid_loss = self.validation(model, epoch, dev_loader)
                # If best accuracy so far, save model as best and the accuracy
                if curr_valid_loss < best_valid_loss:
                    self.logger.info("New best loss, Model saved")
                    is_best = True
                    best_valid_loss = curr_valid_loss
                    best_valid_epoch = epoch
                    epochs_no_improve = 0
                    torch.save(model, './model_dict/model.pt')
            if epochs_no_improve > 3:
                self.logger.info("No best dev loss, stop training.")
                break

    def validation(self, model, epoch, dev_loader):
        """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
        model = model.to(self.rank)
        model.eval()

        running_loss = 0
        correct_words = 0
        total_words = 0
        total_num = 0
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                src_input_ids, src_input_masks, src_type_ids = batch[0].to(self.rank), batch[1].to(self.rank), batch[
                    5].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[
                    4].to(self.rank)

                # Compute output of model
                output = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks, src_type_ids)

                # Get model predictions
                predictions = output.topk(1)[1].squeeze()

                # Compute loss
                loss, n_correct, n_word = self.cal_performance(output, trg_ground_ids, smoothing=True)
                correct_words += n_correct
                total_words += n_word
                # -------------
                running_loss += loss.item()
                total_num += len(batch[0])
        # print statistics
        final_loss = running_loss / (i + 1)

        accuracy = float(100.0 * correct_words) / total_words
        self.logger.info(f"Validation. Epoch: {epoch + 1}, avg dev loss: {final_loss:.4f}, accuracy: {accuracy:.2f}%")

        # return accuracy
        return final_loss

    def test(self, out_max_len=64):
        test_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        model = torch.load('./model_dict/model.pt').to(self.rank)
        f = open(os.path.join(self.dataset_dir, 'results.csv'), 'a+', encoding='utf-8')

        generated_token = []
        gold_token = []
        for batch in tqdm(test_loader):
            src_input_ids, src_input_masks, src_type_ids = batch[0].to(self.rank), batch[1].to(self.rank), batch[
                5].to(self.rank)
            trg_ground_ids = batch[3].to(self.rank)
            per_input_ids, query_input_ids = batch[6].to(self.rank), batch[7].to(self.rank)

            memory = model.encode(src_input_ids, src_input_masks, src_type_ids).transpose(0, 1)
            tgt_input_ids = torch.zeros(src_input_ids.shape[0], self.tgt_len, dtype=torch.long, device=self.rank)
            tgt_input_ids[:, 0] = self.cls_id  # bert sentence head
            for j in range(1, out_max_len):
                tgt_input_masks = torch.zeros(src_input_ids.shape[0], self.tgt_len, dtype=torch.long, device=self.rank)
                tgt_input_masks[:, :j] = 1

                src_attention_masks = ((1 - src_input_masks) > 0)
                tgt_attention_masks = ((1 - tgt_input_masks) > 0)

                output = model.decode(memory, tgt_input_ids[:, :j], src_input_ids, tgt_attention_masks[:, :j],
                                      src_attention_masks)
                _, ids = output.topk(1)
                ids = ids.squeeze(-1)

                tgt_input_ids[:, j] = ids[:, -1]

                if ids[:, -1] == self.sep_id:
                    break
            string = self.decode(tgt_input_ids)[0]
            if len(string) == 0:
                string = self.decode(src_input_ids)[0]
            pred_string = re.sub(r"\s{1,}", "", string)

            per_string = self.decode(per_input_ids)[0]
            per_string = re.sub(r"\s{1,}", "", per_string)
            query_string = self.decode(query_input_ids)[0]
            query_string = re.sub(r"\s{1,}", "", query_string)
            trg_string = self.decode(trg_ground_ids)[0]
            trg_string = re.sub(r"\s{1,}", "", trg_string)

            generated_token += pred_string
            gold_token += trg_string

            f.write(f"persona: {per_string[:150]}\nquery: {query_string[:100]}\ngold: {trg_string[:100]}\nresponse: {pred_string[:100]}\n")

        bleu_1, bleu_2, bleu_3, bleu_4, F1, hyp_d1, hyp_d2, ref_d1, ref_d2 = self.automated_metrics(generated_token, gold_token)
        f.write('BLEU 1-gram: %f\n' % bleu_1)
        f.write('BLEU 2-gram: %f\n' % bleu_2)
        f.write('BLEU 3-gram: %f\n' % bleu_3)
        f.write('BLEU 4-gram: %f\n' % bleu_4)
        f.write('F1-score: %f\n' % F1)
        f.write(f"Distinct-1 (hypothesis, reference): {round(hyp_d1, 4)}, {round(ref_d1, 4)}\n")
        f.write(f"Distinct-2 (hypothesis, reference): {round(hyp_d2, 4)}, {round(ref_d2, 4)}\n")
        f.close()

    def greedy_decode(self, model, src_seq, src_mask, out_max_len=64):
        model.eval()

        with torch.no_grad():
            memory = model.encode(src_seq, src_mask).transpose(0, 1)
            dec_seq = torch.full((src_seq.size(0),), self.cls_id).unsqueeze(-1).type_as(src_seq)

            src_attention_masks = ((1 - src_mask) > 0)
            for i in range(out_max_len):
                dec_output = model.decode(memory, dec_seq, src_seq, None, src_attention_masks)
                dec_output = dec_output.max(-1)[1]
                dec_seq = torch.cat((dec_seq, dec_output[:, -1].unsqueeze(-1)), 1)
        return dec_seq

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def generate(self, out_max_length=64, top_k=40, top_p=0.9, max_length=200):
        test_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        model = torch.load('./model_dict/model.pt').to(self.rank)
        f = open(os.path.join(self.dataset_dir, 'top_k_results.csv'), 'a+', encoding='utf-8')

        generated_token = []
        gold_token = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                src_input_ids, src_input_masks, src_type_ids = batch[0].to(self.rank), batch[1].to(self.rank), batch[
                    5].to(self.rank)
                trg_ground_ids = batch[3].to(self.rank)
                per_input_ids, query_input_ids = batch[6].to(self.rank), batch[7].to(self.rank)

                memory = model.encode(src_input_ids, src_input_masks, src_type_ids).transpose(0, 1)
                tgt_input_ids = torch.zeros(src_input_ids.shape[0], self.tgt_len, dtype=torch.long, device=self.rank)
                tgt_input_ids[:, 0] = self.cls_id  # bert sentence head
                output_ids = []
                for j in range(1, out_max_length):
                    tgt_input_masks = torch.zeros(src_input_ids.shape[0], self.tgt_len, dtype=torch.long,
                                                  device=self.rank)
                    tgt_input_masks[:, :j] = 1

                    src_attention_masks = ((1 - src_input_masks) > 0)
                    tgt_attention_masks = ((1 - tgt_input_masks) > 0)

                    scores = model.decode(memory, tgt_input_ids[:, :j], src_input_ids, tgt_attention_masks[:, :j],
                                          src_attention_masks)

                    logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                    logit_score[self.unk_id] = -float('Inf')

                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                    for id_ in set(output_ids):
                        logit_score[id_] /= 2.0

                    filtered_logits = self.top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    if self.sep_id == next_token.item():
                        break
                    tgt_input_ids[:, j] = next_token.item()
                    output_ids.append(next_token.item())
                string = self.tokenizer.decode(torch.tensor(output_ids))

                pred_string = re.sub(r"\s{1,}", "", string)
                per_string = self.decode(per_input_ids)[0]
                per_string = re.sub(r"\s{1,}", "", per_string)
                query_string = self.decode(query_input_ids)[0]
                query_string = re.sub(r"\s{1,}", "", query_string)
                trg_string = self.decode(trg_ground_ids)[0]
                trg_string = re.sub(r"\s{1,}", "", trg_string)

                generated_token += pred_string
                gold_token += trg_string

                f.write(f"persona: {per_string[:150]}\nquery: {query_string[:100]}\ngold: {trg_string[:100]}\nresponse: {pred_string[:100]}\n")

            bleu_1, bleu_2, bleu_3, bleu_4, F1, hyp_d1, hyp_d2, ref_d1, ref_d2 = self.automated_metrics(generated_token,gold_token)
            f.write('BLEU 1-gram: %f\n' % bleu_1)
            f.write('BLEU 2-gram: %f\n' % bleu_2)
            f.write('BLEU 3-gram: %f\n' % bleu_3)
            f.write('BLEU 4-gram: %f\n' % bleu_4)
            f.write('F1-score: %f\n' % F1)
            f.write(f"Distinct-1 (hypothesis, reference): {round(hyp_d1, 4)}, {round(ref_d1, 4)}\n")
            f.write(f"Distinct-2 (hypothesis, reference): {round(hyp_d2, 4)}, {round(ref_d2, 4)}\n")
            f.close()

    def eval(self):
        """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
        model = torch.load('./model_dict/model.pt').to(self.rank)
        model.eval()

        data_loader = self.make_dataloader(0, self.test_data, 1, shuffle=False)

        f = open(os.path.join(self.dataset_dir, 'eval_results.csv'), 'a+', encoding='utf-8')

        running_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                src_input_ids, src_input_masks, src_type_ids = batch[0].to(self.rank), batch[1].to(self.rank), batch[
                    5].to(self.rank)
                trg_input_ids, trg_ground_ids, trg_input_masks = batch[2].to(self.rank), batch[3].to(self.rank), batch[
                    4].to(self.rank)
                per_input_ids, query_input_ids = batch[6].to(self.rank), batch[7].to(self.rank)

                # Compute output of model
                output = model(src_input_ids, src_input_masks, trg_input_ids, trg_input_masks, src_type_ids)

                # Get model predictions
                predictions = output.topk(1)[1].squeeze()
                predictions = predictions * trg_input_masks

                # Compute loss
                loss, n_correct, n_word = self.cal_performance(output, trg_ground_ids, smoothing=False)
                accuracy = float(100.0 * n_correct) / n_word
                running_loss += loss.item()

                # print predict
                per_string = self.decode(per_input_ids)[0]
                per_string = re.sub(r"\s{1,}", "", per_string)
                query_string = self.decode(query_input_ids)[0]
                query_string = re.sub(r"\s{1,}", "", query_string)
                trg_string = self.decode(trg_input_ids)[0]
                trg_string = re.sub(r"\s{1,}", "", trg_string)
                pred_string = self.decode(predictions)[0]
                pred_string = re.sub(r"\s{1,}", "", pred_string)

                # f.write(src_string + '\t' + trg_string + '\t' + pred_string + '\n')
                print(f"persona: {per_string[:150]}\nquery: {query_string[:100]}\ngold: {trg_string[:100]}\nresponse: {pred_string[:100]}\n")
                f.write(f"persona: {per_string[:150]}\nquery: {query_string[:100]}\ngold: {trg_string[:100]}\nresponse: {pred_string[:100]}\n")

        loss = running_loss / len(data_loader)
        ppl = math.exp(loss)
        print(f"eval loss: {loss:.4f}, ppl: {ppl:.2f}, accuracy: {accuracy:.2f}%")
        f.write(f"eval loss: {loss:.4f}, ppl: {ppl:.2f}, accuracy: {accuracy:.2f}%")

        f.close()


    def automated_metrics(self, generated_token, gold_token):
        # bleu-1和bleu-2
        assert len(generated_token) == len(gold_token)
        reference = []
        candidate = []
        for i in range(0, len(generated_token)):
            reference.append([self.tokenizer.tokenize(gold_token[i])])
            candidate.append(self.tokenizer.tokenize(generated_token[i]))
        bleu_1 = corpus_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        print('BLEU 1-gram: %f' % bleu_1)
        print('BLEU 2-gram: %f' % bleu_2)
        print('BLEU 3-gram: %f' % bleu_3)
        print('BLEU 4-gram: %f' % bleu_4)

        # F1-score
        assert len(generated_token) == len(gold_token)
        F1 = 0
        for i in range(0, len(generated_token)):
            reference = tokenizer.tokenize(generated_token[i])
            candidate = tokenizer.tokenize(gold_token[i])

            c = 0
            r_list = []
            for j in range(0, len(candidate)):
                for k in range(0, len(reference)):
                    if candidate[j] == reference[k] and len(r_list) == 0:
                        c += 1
                        r_list.append(k)
                        break
                    false_num = 0
                    for s in [k != rl for rl in r_list]:
                        if not s:
                            false_num += 1
                    if candidate[j] == reference[k] and false_num < 1:
                        c += 1
                        r_list.append(k)
                        break

            r = 0
            c_list = []
            for j in range(0, len(reference)):
                for k in range(0, len(candidate)):
                    if reference[j] == candidate[k] and len(c_list) == 0:
                        r += 1
                        c_list.append(k)
                        break
                    false_num = 0
                    for s in [k != cl for cl in c_list]:
                        if not s:
                            false_num += 1
                    if reference[j] == candidate[k] and false_num < 1:
                        r += 1
                        c_list.append(k)
                        break

            precision = c / len(candidate)
            recall = r / len(reference)
            if (precision + recall) == 0:
                F1 = F1 + 0
            else:
                F1 = F1 + 2 * (precision * recall) / (precision + recall)

        F1 = F1 / len(generated_token)
        print('F1-score: %f' % F1)

        # distinct-1和distinct-2
        hyp_d1, hyp_d2 = eval_distinct(generated_token)
        ref_d1, ref_d2 = eval_distinct(gold_token)

        print(f"Distinct-1 (hypothesis, reference): {round(hyp_d1, 4)}, {round(ref_d1, 4)}")
        print(f"Distinct-2 (hypothesis, reference): {round(hyp_d2, 4)}, {round(ref_d2, 4)}")

        return bleu_1, bleu_2, bleu_3, bleu_4, F1, hyp_d1, hyp_d2, ref_d1, ref_d2
