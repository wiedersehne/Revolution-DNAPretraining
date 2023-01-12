import torch
import pyfaidx
import kipoiseq
import numpy
import numpy as np
from kipoiseq import Interval
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer

fasta_file = './data/hg38.fa'


class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


fasta_extractor = FastaStringExtractor(fasta_file)
hg38_dict = SeqIO.to_dict(SeqIO.parse("./data/hg38.fa", "fasta"))
gene = ['a','t','c','g','n']
lb = LabelBinarizer()
lb.fit(list(gene))


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def vcf_one(chr, pos, ref, alt, length):
    # variant = kipoiseq.Variant('chr16', 57025062, 'C', 'T')  # @param
    variant = kipoiseq.Variant(chr, pos, ref, alt)
    interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(length)
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    # print(reference)
    reference_code = np.array(lb.transform(list(reference.lower())))
    alternate_code = np.array(lb.transform(list(alternate.lower())))
    # print(reference_code[0])
    return reference_code.T, alternate_code.T


def vcf_many(many, length):
    ref_all = []
    alt_all = []
    tissue_all = []
    label_all = []
    for _, seq in many.iterrows():
        ref, alt = vcf_one(seq['chr'], seq['pos'], seq['ref'], seq['alt'], length)
        tissue = seq['tissue']
        label = seq['label']
        ref_all.append(ref)
        alt_all.append(alt)
        tissue_all.append(tissue)
        label_all.append(label)

    ref_all = torch.from_numpy(numpy.array(ref_all))
    alt_all = torch.from_numpy(numpy.array(alt_all))
    tissue_all = torch.from_numpy(numpy.array(tissue_all))
    label_all = torch.from_numpy(numpy.array(label_all))
    return ref_all, alt_all, tissue_all, label_all


def get_refs(many, length):
    ref_all = []
    for _, seq in many.iterrows():
        ref, alt = vcf_one(seq['chr'], seq['pos'], seq['ref'], seq['alt'], length)
        ref_all.append(ref.astype(numpy.int8))
    return ref_all


def get_refs_batch(many, length):
    ref_all = []
    for _, seq in many.iterrows():
        ref, alt = vcf_one(seq['chr'], seq['pos'], seq['ref'], seq['alt'], length)
        ref_all.append(ref)
        
    return ref_all

