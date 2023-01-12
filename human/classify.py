import os
# import sys
import wandb
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *
# from chord_mixer import LeiModel_sig_avg as LeiModel
# from models import Classifier
from torch.utils.data import Dataset
from vcf_encode import *
from model_former_ende import *
from torch.optim.lr_scheduler import OneCycleLR
# torch.autograd.set_detect_anomaly(True)


# evaluation for genomde variant classification
def net_eval(net, criterion, ep, name, eval_loader, best_losses, best_roc, out, length, device):
    OUT = out
    eval_losses = 0
    eval_preds = [[] for _ in range(OUT)]
    eval_labels = [[] for _ in range(OUT)]
    eval_rocs = []
    eval_aps = []

    eval_start = datetime.now()
    for eval_ref, eval_alt, eval_tissue, eval_label in eval_loader:
        eval_ref, eval_alt, eval_tissue, eval_label = eval_ref.to(device), eval_alt.to(device), eval_tissue.to(device), eval_label.to(device)
        eval_pred = net(eval_ref, eval_alt, eval_tissue)
        eval_loss = criterion(eval_pred, eval_label)
        eval_losses += eval_loss

        for t, y, l in zip(eval_tissue, eval_pred, eval_label):
            eval_preds[t.item()].append(y.item())
            eval_labels[t.item()].append(l.item())

    for eval_i in range(OUT):
        eval_rocs.append(roc_auc_score(eval_labels[eval_i], eval_preds[eval_i]))
        eval_aps.append(average_precision_score(eval_labels[eval_i], eval_preds[eval_i]))

    eval_roc = np.average(eval_rocs)
    eval_ap = np.average(eval_aps)
    eval_losses_mean = eval_losses / len(eval_loader)
    eval_end = datetime.now()
    eval_time = (eval_end - eval_start).total_seconds()

    if eval_losses_mean > best_losses:
        print_loss = '{} loss'
    else:
        best_losses = eval_losses_mean
        print_loss = 'best {} loss'
    if eval_roc < best_roc:
        print_roc = '{} roc'
    else:
        best_roc = eval_roc
        df = pd.DataFrame(eval_rocs)
        df.to_csv(f"test_rocs_{length}.csv")
        print_roc = 'best {} roc'
    print_epoch = 'Epoch {}, ' + print_loss + ': {}, task: {}, ' + print_roc + ': {}, ap: {}, Time: {}'
    wandb.log({name + " loss": eval_losses_mean})
    wandb.log({name + " roc": best_roc})

    return best_losses, best_roc


# Predict genome variant using pretrained model
def classify(cfg, pretrain_cfg):

    # setting
    device = cfg.device
    KS = cfg.kernel_size
    IN_dim = cfg.dim
    OUT = cfg.n_class

    LAYER = cfg.n_layers
    DIM = cfg.hdim1
    DIM2 = cfg.hdim2
    LR = cfg.lr
    BATCH = cfg.batch_size
    EPOCH = cfg.n_epochs
    LENGTH = cfg.max_len
    # load the model
    print("loading the pretrained model")
    # net = Classifier(IN_dim, DIM, DIM2, LAYER, KS, OUT, LENGTH)
    # model, depth, heads, input_size, hdim1, hdim2, n_targets, seq_len
    net = Classifier("nystromer", 4, 4, IN_dim, DIM, DIM2, OUT, LENGTH)
    print(net)
    # load_cdil_head(net, f'./exp/pretrain/classify_ftcnn_freeze_steps_25_mask_rate_0.3_mask_ratio_0.8_length_1000_dim_16.pt')
    # load_same(net, f'./exp/pretrain/model_ve_1hot_steps_{pretrain_cfg.n_epochs-1}_mask_rate_{pretrain_cfg.mask_ratio}_mask_ratio_{pretrain_cfg.real_mask}_len_{LENGTH}_dim1_{pretrain_cfg.hdim1}_dim2_{pretrain_cfg.hdim2}.pt')
    # load_same(net, f'./exp/pretrain/model_ve_1hot_steps_49_mask_rate_0.3_mask_ratio_0.8_len_3000_dim1_32_dim2_16.pt')
    net = nn.DataParallel(net, device_ids = [0,1,2,3])#,2,3,4,5,6,7])
    net = net.to(device)
    # optimization
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=1e-6, momentum=0.9)
    best_val_losses = 999999.
    best_test_losses = 999999.
    best_val_roc = 0.
    best_test_roc = 0.
    # dataset
    print("Preparing dataset")
    train_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_train.csv'), LENGTH), batch_size=BATCH, shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_valid.csv'), LENGTH), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)
    test_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_test.csv'), LENGTH), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=1e-4, epochs=50, steps_per_epoch=len(train_loader))

    # training process
    print("Start training")
    for epoch in range(EPOCH):
        net.train()
        train_losses = 0
        train_preds = [[] for _ in range(OUT)]
        train_labels = [[] for _ in range(OUT)]
        train_rocs = []
        train_aps = []
        t_start = datetime.now()
        for ref, alt, tissue, label in train_loader:
            ref, alt, tissue, label = ref.to(device), alt.to(device), tissue.to(device), label.to(device)
            optimizer.zero_grad()
            pred = net(ref, alt, tissue)
            batch_loss = criterion(pred, label)
            batch_loss.backward()
            optimizer.step()
            train_losses += batch_loss

            for one_tissue, one_y, one_label in zip(tissue, pred, label):
                train_preds[one_tissue.item()].append(one_y.item())
                train_labels[one_tissue.item()].append(one_label.item())

        for i in range(OUT):
            train_rocs.append(roc_auc_score(train_labels[i], train_preds[i]))
            train_aps.append(average_precision_score(train_labels[i], train_preds[i]))

        # print(train_labels[0], train_preds[0])

        train_roc = np.average(train_rocs)
        train_ap = np.average(train_aps)
        train_losses_mean = train_losses / len(train_loader)

        wandb.log({"train loss": train_losses_mean})
        wandb.log({"train roc": train_roc})
        # wandb.log({"train ap": train_ap})

        # save('./exp/pretrain', net, epoch, pretrain_cfg.mask_ratio, pretrain_cfg.real_mask, LENGTH, pretrain_cfg.hdim1)

        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        print(f'train_loss_{train_losses_mean}_train_roc_{train_roc}_time_used_{epoch_time}')

        df = pd.DataFrame(train_rocs)
        df.to_csv(f"train_rocs_{LENGTH}.csv")
        
        with torch.no_grad():
            net.eval()
            best_val_losses, best_val_roc = net_eval(net, criterion, epoch, 'val', val_loader, best_val_losses, best_val_roc, OUT, LENGTH, device)
            best_test_losses, best_test_roc = net_eval(net, criterion, epoch, 'test', test_loader, best_test_losses, best_test_roc, OUT, LENGTH, device)
            print(f'val_loss_{best_val_losses}_val_roc_{best_val_roc}')
            print(f'test_loss_{best_test_losses}_test_roc_{best_test_roc}')
