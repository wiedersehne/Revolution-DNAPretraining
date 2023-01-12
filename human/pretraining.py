print("Inside python")
from random import randint, shuffle
from random import random as rand
from utils import *
from classify import classify
import pandas as pd
import torch.nn as nn
import sys
#import chord_mixer as models
import models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import train_hg38 as train
import fire
from vcf_encode import *
import random
import copy
from pyfaidx import Fasta
from pyfaidx import Faidx
from multiprocessing import Pool
import multiprocessing as mp
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import wandb
from model_former_ende import *
print("initializing wandb")
wandb.init(settings=wandb.Settings(start_method='fork'), project="Xformer", entity="tonyu")
print("wandb initialized")
torch.backends.cudnn.bechmark = True

#393216
pretrain_cfg = {
    "seed": 3431,
    "dim": 5,
    "hdim1":16,
    "hdim2":8,
    "kernel_size": 3,
    "n_layers": 9,
    "max_len": 1000,
    "batch_size": 64,
    "mask_ratio": 0.3,
    "real_mask": 0.8,
    "lr": 0.003,
    "n_epochs": 50,
    "train_num":40000,
    "n_cores": mp.cpu_count(),
    "device": "cuda"
}

classify_cfg = {
    "dim": 5,
    "hdim1":16,
    "hdim2":8,
    "kernel_size": 3,
    "n_layers": 9,
    "max_len": 1000,
    "n_class": 49,
    "batch_size": 8,
    "lr": 0.00003,
    "n_epochs": 50,
    "device": "cuda"
}

class Dict2Class:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

print(pretrain_cfg)
print(classify_cfg)
pretrain_cfg = Dict2Class(pretrain_cfg)
classify_cfg = Dict2Class(classify_cfg)


class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, gene_labels, masked_gene, masked_pos, sequences_num, ids, pos):
        self.data = gene_labels
        self.masked_data = masked_gene
        self.masked_pos = masked_pos
        self.sequences_num = sequences_num
        self.ids = ids
        self.pos = pos

    def __getitem__(self, index):
        """
        Returns: tuple (sequence, masked_tokens, masked_position)
        """
        randid = random.randint(0, 78333)
        # print("index",index)
        # print(self.ids[index][0:2])
        # print(self.ids[index][3:])
        random_chrom = int(self.ids[randid][3:])
        masked_gene, gene_labels, masked_vec = self.masked_data[random_chrom], self.data[random_chrom], self.masked_pos[random_chrom]
        # length = len(gene_labels)
        # cursor = random.randint(0, length-pretrain_cfg.max_len)
        cursor = self.pos[randid]
        halflength = int(pretrain_cfg.max_len/2)
        if cursor + halflength >= len(gene_labels):
            cursor = 20000
        X = torch.from_numpy(masked_gene[cursor-halflength:cursor+halflength])
        P = torch.from_numpy(masked_vec[cursor-halflength:cursor+halflength])
        T = torch.from_numpy(gene_labels[cursor-halflength:cursor+halflength])
        # print(random_chrom, cursor, X.shape)
        return (X, T, P)

    def __len__(self):
        return self.sequences_num


class UniformMasking():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_prob, real_mask_ratio):
        super().__init__()
        self.mask_ratio = mask_prob
        self.real_mask_ratio = real_mask_ratio

    def __call__(self, instance):
        # print(instance[0:10])
        # 1. get binary-encoded masked indexes and masked positions
        # random_masked_ratio = (1 - self.real_mask_ratio) / 2
        uniform_vec = np.random.rand(len(instance))
        uniform_vec = uniform_vec <= self.mask_ratio
        masked_vec = uniform_vec.astype(int)
        
        # 2. get real and random binary-encoded masked indexes
        uniform_vec2 = np.random.rand(len(instance))
        random_vec = np.zeros(len(instance))
        same_vec = np.zeros(len(instance))
        random_vec[(masked_vec == 1) & (uniform_vec2 <= 0.1)] = 1
        same_vec[(masked_vec == 1) & (uniform_vec2 >= 0.9)] = 1
        real_vec = abs(masked_vec - random_vec - same_vec)
        random_vec = np.array(random_vec).astype(bool)
        real_vec = np.array(real_vec).astype(bool)

        # 3. masking with all zeros.
        instance[real_vec,:] = [0, 0, 0, 0, 0]
        # 4. masking with random one-hot encode
        instance[random_vec,:] = np.eye(5)[np.random.choice(5, 1)]

        return instance, masked_vec


# Model for Pretraining
# class Model4Pretrain(nn.Module):
#     "CDIL Model for Pretrain : Masked LM"
#     def __init__(self, cfg):
#         super().__init__()
#         self.cdilNet = models.CDIL_Conv(cfg.dim, [cfg.hdim1]*cfg.n_layers, cfg.hdim1*2, cfg.kernel_size)
#         # self.cdilNet2 = models.CDIL_Conv(cfg.hdim1, [cfg.hdim1]*cfg.n_layers, cfg.hdim1, cfg.kernel_size)
#         # self.cdilNet = models.ChordMixerNet(cfg.dim, 12, cfg.max_len, [128, "GELU"], 0)
#         self.hidden_list = [cfg.hdim2]*cfg.n_layers
#         self.hidden_list[-1] = cfg.dim
#         self.decoder = models.CDIL_Conv(cfg.hdim1, self.hidden_list, cfg.hdim2*2, cfg.kernel_size)

#     def forward(self, input_seq):
#         input_seq = input_seq.float()
#         # encoder
#         h = torch.permute(input_seq, (0, 2, 1))
#         h = self.cdilNet(h)
#         # decoder
#         h = self.decoder(h)
#         h = torch.permute(h, (0, 2, 1))

#         return h


def masking_one_chrom(gene_id):
    print("started ")
    gene = list(hg38_dict[gene_id].lower())
    # gene = [i for i in gene if i != 'n']
    gene_label = le.transform(copy.deepcopy(gene))
    # gene_label = lb.transform(copy.deepcopy(gene))
    masking = UniformMasking(pretrain_cfg.mask_ratio, pretrain_cfg.real_mask)
    masked_gene, masked_vec = masking(np.array(lb.transform(gene)))
    print("done")
    return gene_label, masked_gene, masked_vec


def parallel_process(lb, le):
    hg38_dict = SeqIO.to_dict(SeqIO.parse("./data/hg38.fa", "fasta"))
    hg38_length = pd.read_csv('./data/hg38_length.csv')
    lb.fit(['a','t','c','g','n'])
    le.fit(['a','t','c','g','n'])
    chromosome_ids = hg38_length.name.values
    print("start multiprocessing")
    with Pool(pretrain_cfg.n_cores) as pool:
        result = pool.map(masking_one_chrom, chromosome_ids)
    gene_labels, masked_genes, masked_vecs = map(list, zip(*result))
    masked_gene = np.array(masked_genes)
    masked_vec = np.array(masked_vecs)
    # print(gene_labels[0], masked_gene[0], masked_vec[0])
    # print(len(masked_gene), len(gene_labels), len(masked_vec))
    return gene_labels, masked_gene, masked_vec


if __name__ == '__main__':
    set_seeds(pretrain_cfg.seed)
    device = pretrain_cfg.device
    data_parallel = False
    save_dir='./exp/pretrain/'
    # 1. preprocessing data
    lb = LabelBinarizer()
    le = LabelEncoder()
    print("preprocessing-------")
    gene_label, maksed_gene, masked_pos = parallel_process(lb, le)#"chr19_KI270938v1_alt")
    # for x in maksed_gene:
    #     print(len(x))
    train_df = pd.read_csv('./data/vcf_train.csv')
    ids = train_df.chr.values
    pos = train_df.pos.values
    print(len(ids))
    print(ids[0:50])
    print("building dataloader-------")
    # 2. build data loader
    train_data_iter = DataLoader(
                        DatasetCreator(gene_label, maksed_gene, masked_pos, pretrain_cfg.train_num, ids, pos),
                        batch_size=pretrain_cfg.batch_size,
                        shuffle=True,
                        drop_last=False,
                        num_workers=1,
                        pin_memory=True
                    )

    print(len(train_data_iter.dataset))

    valid_data_iter = DataLoader(
                        DatasetCreator(gene_label, maksed_gene, masked_pos, int(pretrain_cfg.train_num*0.125), ids, pos),
                        batch_size=pretrain_cfg.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=1,
                        pin_memory=True
                    )

    test_data_iter = DataLoader(
                        DatasetCreator(gene_label, maksed_gene, masked_pos, int(pretrain_cfg.train_num*0.125), ids, pos),
                        batch_size=pretrain_cfg.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=1,
                        pin_memory=True
                    )

    print("training----------")
    model = Model4Pretrain(pretrain_cfg)
    # model = Model4Pretrain("nystromer", 2, 2, 5, 16, 8)
    print(model)
    # #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss(reduce=False)
    # criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = pretrain_cfg.lr)
    trainer = train.Trainer(pretrain_cfg, model, train_data_iter, valid_data_iter, test_data_iter,
                            optimizer, criterion, save_dir, device)
    trainer.train(data_parallel)

    classify(classify_cfg, pretrain_cfg)
