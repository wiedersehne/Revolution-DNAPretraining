import os
import json
from typing import NamedTuple
import torch
import torch.nn as nn
import time
import wandb


class Trainer(object):
    """Training Helper Class"""

    def __init__(self, cfg, model, data_iter, val_data_iter, test_data_iter, optimizer, criterion, save_dir, device):
        self.cfg = cfg  # config for training : see class Config
        self.model = model
        self.data_iter = data_iter  # iterator to load data
        self.valid_data_iter = val_data_iter
        self.test_data_iter = test_data_iter
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device  # device name
        self.criterion = criterion
        self.mask_rate = cfg.mask_ratio
        self.real_mask = cfg.real_mask
        self.hdim = cfg.hdim1
        self.hdim2 = cfg.hdim2
        self.maxlen = cfg.max_len

    def train(self, data_parallel=False):
        """ Train Loop """
        self.model.train()  # train mode
        if data_parallel:  # use Data Parallelism with Multi-GPU
            self.model = nn.DataParallel(self.model, device_ids=[0,1])#,2,3,4,5,6,7])

        self.model = self.model.to(self.device)

        for e in range(self.cfg.n_epochs):
            time_s = time.time()
            loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
            iter_bar = self.data_iter
            correct_train = 0
            total_train = 0
            for i, (input_seq, tokens, masked_pos) in enumerate(iter_bar):
                input_seq, tokens, masked_pos = input_seq.to(self.device), tokens.to(self.device), masked_pos.to(self.device)
                for p in self.model.parameters():
                    p.grad = None

                logits_lm = self.model(input_seq)
                predicted = logits_lm
                _, predicted = logits_lm.max(2)
                predicted = predicted[masked_pos==1]
                masked_tokens = tokens[masked_pos==1]
                # print("predicted", predicted.shape)
                # print("masked tokens", masked_tokens.shape)
                loss_lm = self.criterion(logits_lm.transpose(1, 2), tokens)
                # print("loss_lm", loss_lm.shape)
                non_zeros = len(masked_pos[masked_pos==1])
                loss_lm = loss_lm * masked_pos
                loss_lm = loss_lm.sum()/non_zeros
                loss_lm = loss_lm.float()
                loss_lm.backward()
                loss_sum += loss_lm.item()
                # correct_train += predicted.eq(masked_tokens).sum().item()
                # total_train += len(masked_tokens)
                self.optimizer.step()


            if e == self.cfg.n_epochs-1:  # save
                print("Saving a new model------")
                self.save(e)

            time_e = time.time()
            wandb.log({"train_loss": loss_sum / len(iter_bar)})

            if self.cfg.n_epochs % 1 == 0:
                self.model.eval()
                val_iter_bar = self.valid_data_iter
                test_iter_bar = self.test_data_iter
                val_loss_sum = 0.0
                test_loss_sum = 0.0
                correct_val = 0
                correct_test = 0
                total_val = 0
                total_test = 0
                with torch.no_grad():
                    time_s = time.time()
                    for i, (input_seq, tokens, masked_pos) in enumerate(val_iter_bar):
                        input_seq, tokens, masked_pos = input_seq.to(self.device), tokens.to(self.device), masked_pos.to(self.device)
                        logits = self.model(input_seq)
                        _, predicted = logits.max(2)
                        predicted = logits
                        predicted = predicted[masked_pos==1]
                        masked_tokens = tokens[masked_pos==1]
                        val_loss = self.criterion(logits.transpose(1, 2), tokens) # for masked LM
                        non_zeros = len(masked_pos[masked_pos==1])
                        val_loss = val_loss * masked_pos
                        val_loss = val_loss.sum()/non_zeros
                        val_loss = val_loss.float()
                        # correct_val += predicted.eq(masked_tokens).sum().item()
                        # total_val += len(masked_tokens)
                        val_loss = val_loss.mean()
                        val_loss_sum += val_loss.item()
                    time_e = time.time()
                    wandb.log({"val_loss": val_loss_sum / len(val_iter_bar)})
                    # wandb.log({"val_acc": correct_val / total_val})

                    # print('Epoch %d/%d : Average Val LOSS %5.3f, Val_ACC %5.3f, Used Time %5.3f' % (
                    # e + 1, self.cfg.n_epochs, val_loss_sum / (i + 1), 100*correct_val/total_val, time_e - time_s))

                time_s = time.time()
                for i, (input_seq, tokens, masked_pos) in enumerate(test_iter_bar):
                    #input_seq, masked_tokens, masked_pos = torch.Tensor(input_seq).to(torch.int64), torch.Tensor(masked_tokens).to(torch.int64), torch.Tensor(masked_pos).to(torch.int64)
                    input_seq, tokens, masked_pos = input_seq.to(self.device), tokens.to(self.device), masked_pos.to(self.device)
                    logits = self.model(input_seq)
                    # _, predicted = logits.max(2)
                    predicted = logits
                    predicted = predicted[masked_pos==1]
                    masked_tokens = tokens[masked_pos==1]
                    test_loss = self.criterion(logits.transpose(1, 2), tokens)  # for masked LM
                    # print(loss_lm.shape)
                    non_zeros = len(masked_pos[masked_pos==1])
                    test_loss = test_loss * masked_pos
                    test_loss = test_loss.sum()/non_zeros
                    # correct_test += predicted.eq(masked_tokens).sum().item()
                    test_loss = test_loss.float()
                    total_test += len(masked_tokens)
                    test_loss = test_loss.mean()
                    test_loss_sum += test_loss.item()
                time_e = time.time()
                wandb.log({"test_loss": test_loss_sum / len(test_iter_bar)})

    def save(self, i):
        """ save current model """
        torch.save(self.model.encoder.state_dict(),  # save model object before nn.DataParallel
                   os.path.join(self.save_dir, 'model_ve_1hot_steps_' + str(i) + '_mask_rate_' + str(self.mask_rate) + '_mask_ratio_' + str(self.real_mask) + '_len_' + str(self.maxlen) + '_dim1_' + str(self.hdim) + '_dim2_' + str(self.hdim2) + '.pt'))
