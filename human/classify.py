import os
import wandb
import torch
import argparse
import yaml
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *
from revolution_models import Classifier
from torch.utils.data import Dataset
from vcf_encode import *
from model_former_ende import *
wandb.init(settings=wandb.Settings(start_method='fork'), project="fine-tuning", entity="tonyu")

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='Fine-tuning')
parser.add_argument("--model", type=str, default='Revolution')
args = parser.parse_args()

# Parsing training config
stream = open("config.yaml", 'r')
cfg_yaml = yaml.safe_load(stream)
config = cfg_yaml[args.problem][args.model]
pretrain_cfg = cfg_yaml["Pretraining"][args.model]
print('pretraining config', pretrain_cfg)
print('model_config', config)


def net_eval(net, criterion, epoch, name, data_loader, tissues, length, device):
    tissues = tissues
    best_losses = 9999999
    best_roc = 0.0
    eval_losses = 0.0
    eval_preds = [[] for _ in range(OUT)]
    eval_labels = [[] for _ in range(OUT)]
    eval_rocs = []
    eval_aps = []
    eval_start = datetime.now()

    for eval_ref, eval_alt, eval_tissue, eval_label in data_loader:
        eval_ref, eval_alt, eval_tissue, eval_label = eval_ref.to(device), eval_alt.to(device), eval_tissue.to(device), eval_label.to(device)
        eval_pred = net(eval_ref, eval_alt, eval_tissue)
        eval_loss = criterion(eval_pred, eval_label)
        eval_losses += eval_loss

        for t, y, l in zip(eval_tissue, eval_pred, eval_label):
            eval_preds[t.item()].append(y.item())
            eval_labels[t.item()].append(l.item())

    for eval_i in range(tissue):
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
        # df.to_csv(f"test_rocs_{length}.csv")
        print_roc = 'best {} roc'
    print_epoch = 'Epoch {}, ' + print_loss + ': {}, task: {}, ' + print_roc + ': {}, ap: {}, Time: {}'
    wandb.log({name + " loss": eval_losses_mean})
    wandb.log({name + " roc": best_roc})

    return best_losses, best_roc


# Predict genome variant using pretrained model
def classify(cfg, pretrain_cfg):

    # setting
    device = cfg["device"]
    kernel_size = cfg["kernel_size"]
    input_dim = cfg["dim"]
    output_dim = cfg["n_class"]
    n_layers = cfg["n_layers"]
    dim1 = cfg["hdim1"]
    dim2 = cfg["hdim2"]
    lr = cfg["lr"]
    batch = cfg["batch_size"]
    n_epochs = cfg["n_epochs"]
    length = cfg["max_len"]

    if args.model == "Revolution":
        net = Classifier(input_dim, dim1, dim2, n_layers, kernel_size, output_dim, length)
    else:
        net = Former_Classifier(cfg["name"], cfg["n_layers"], cfg[n_heads], input_dim, dim1, dim2, out_dim, length)
    print(net)

    if cfg["pretrain"]:
        print("loading the pretrained model")
        load_same(net, f'./exp/pretrain/model_ve_1hot_steps_{pretrain_cfg["n_epochs"]-1}_mask_rate_{pretrain_cfg["mask_ratio"]}_mask_ratio_{pretrain_cfg["real_mask"]}_len_{length}_dim1_{pretrain_cfg["hdim1"]}_dim2_{pretrain_cfg["hdim2"]}.pt')
    # load_same(net, f'./exp/pretrain/model_ve_1hot_steps_49_mask_rate_0.3_mask_ratio_0.8_len_3000_dim1_32_dim2_16.pt')
    net = nn.DataParallel(net, device_ids = [0,1,2,3])#,2,3,4,5,6,7])
    net = net.to(device)

    # optimization
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
    best_val_losses = 999999.
    best_test_losses = 999999.
    best_val_roc = 0.
    best_test_roc = 0.
    # dataset
    print("Preparing dataset")
    train_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_train.csv'), length), batch_size=batch, shuffle=True, drop_last=False, pin_memory=True)
    val_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_valid.csv'), length), batch_size=batch, shuffle=False, drop_last=False, pin_memory=True)
    test_loader = DataLoader(vcf_Dataset(pd.read_csv('./data/label811_49_test.csv'), length), batch_size=batch, shuffle=False, drop_last=False, pin_memory=True)

    # training process
    print("Start training")
    for epoch in range(n_epochs):
        net.train()
        train_losses = 0
        train_preds = [[] for _ in range(output_dim)]
        train_labels = [[] for _ in range(output_dim)]
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

        for i in range(output_dim):
            train_rocs.append(roc_auc_score(train_labels[i], train_preds[i]))

        train_roc = np.average(train_rocs)
        train_losses_mean = train_losses / len(train_loader)

        wandb.log({"train loss": train_losses_mean})
        wandb.log({"train roc": train_roc})

        # save('./exp/pretrain', net, epoch, pretrain_cfg.mask_ratio, pretrain_cfg.real_mask, LENGTH, pretrain_cfg.hdim1)

        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        print(f'train_loss_{train_losses_mean}_train_roc_{train_roc}_time_used_{epoch_time}')

        df = pd.DataFrame(train_rocs)
        df.to_csv(f"train_rocs_{length}.csv")
        
        with torch.no_grad():
            net.eval()
            best_val_losses, best_val_roc = net_eval(net, criterion, epoch, 'val', val_loader, best_val_losses, best_val_roc, output_dim, length, device)
            best_test_losses, best_test_roc = net_eval(net, criterion, epoch, 'test', test_loader, best_test_losses, best_test_roc, output_dim, length, device)
            print(f'val_loss_{best_val_losses}_val_roc_{best_val_roc}')
            print(f'test_loss_{best_test_losses}_test_roc_{best_test_roc}')

classify(config, pretrain_cfg)
