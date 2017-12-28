from __future__ import print_function
from torch import LongTensor, FloatTensor
import torch.utils.data as data
from torch.autograd import Variable
import os
import errno
import math
from collections import OrderedDict
from utils import *

class WMT(data.Dataset):
    """`WMT News Commentary <http://www.statmt.org/wmt13/>`_ Dataset.

    This loads the WMT europarl-v7.de-en dataset for english-german translation.

    Notes:  In the English set, I found 309 (308 + unknown char) different chars,
        but the paper has only 296.  In the German set, I correctly have 323 chars,
        as used in the paper.

    Args:
        TODO: update documentation
        root (string): Root directory of dataset.
        keep_files(bool, optional): if true, clean up is not performed on downloaded
            files.
    """

    SPLITS = {
        "europarl": ("europarl-v7.de-en.en", "europarl-v7.de-en.de"),
        #"2013_news": ("news-commentary-v8.de-en.en", "news-commentary-v8.de-en.de"),
        #"2014_news": ("news-commentary-v9.de-en.en", "news-commentary-v9.de-en.de"),
        #"2015_news": ("news-commentary-v10.de-en.en", "news-commentary-v10.de-en.de"),
    }

    REPLACE = [("\t", " "), ("—", "–"), ("―", "–"), ("−", "–"), (u'\u200b', ""), (u'\xa0', " ")]

    END = 0 # this is the "\n" character which gets removed with the split op
    UNK = "☺"

    a = 0.2 # from paper a = 1.2
    b = 0 # from paper b = 0

    def __init__(self, root, transform=None, target_transform=None,
                 split="europarl", use_cache=False, randomize=False,
                 download=False, keep_files=False):
        self.root = root
        self.keep_files = keep_files
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        data = {}
        uniq = {}
        for k, v in self.SPLITS.items():
            split_dir = os.path.join(root, k)
            with open(os.path.join(split_dir, v[0]), "r") as f:
                raw = f.read()
                for s, r in self.REPLACE:
                    raw = raw.replace(s, r)
                uniq_en = sorted(list(set(raw)))
                uniq_en = OrderedDict([(k, i) for i, k in enumerate(uniq_en)])
                uniq_en.update({self.UNK: len(uniq_en)})
                english = raw.split("\n")
            with open(os.path.join(split_dir, v[1]), "r") as f:
                raw = f.read()
                for s, r in self.REPLACE:
                    raw = raw.replace(s, r)
                uniq_de = sorted(list(set(raw)))
                uniq_de = OrderedDict([(k, i) for i, k in enumerate(uniq_de)])
                uniq_de.update({self.UNK: len(uniq_de)})
                deutsch = raw.split("\n")
                #deutsch = [[c for c in l] for l in f.readlines()]
                #uniq.update([c for l in deutsch for c in l])
            assert len(english) == len(deutsch)
            data[k] = [(e, d) for e, d in zip(english, deutsch)]
            uniq[k] = (uniq_en, uniq_de)
            #data[k] = [(e, d) for e, d in zip(english, deutsch) \
            #           if (len(d) / len(e) < 1.5 and len(d) / len(e) > 0.3)]
            #print(len(english), len(english) - len(data[k]))

        self.labelers = uniq
        encoded = {}
        for k, v in data.items():
            enc_split = []
            for e, d in v:
                if len(e) > 0:
                    eng_encoded = [self.labelers[k][0][c_e] for c_e in e]
                    eng_encoded += [self.END]
                    eng_encoded += [len(uniq[k][0])] * int(len(eng_encoded) * self.a + self.b)
                    deu_encoded = [self.labelers[k][1][d_e] for d_e in d]
                    deu_encoded += [self.END]
                    enc_split.append((eng_encoded, deu_encoded))
                else:
                    #print("blank found: ", e, d)
                    pass
            encoded[k] = enc_split
        self.data = encoded

        """
        self.labeler = {k: i for i, k in enumerate(sorted(list(uniq)))}
        self.rlabeler = {v: k for k, v in self.labeler.items()}
        encoded = {}
        for k, v in data.items():
            enc_split = []
            for e, d in v:
                eng_encoded = [self.labeler[c_e] for c_e in e]
                deu_encoded = [self.labeler[d_e] for d_e in d]
                enc_split.append((eng_encoded, deu_encoded))
            encoded[k] = enc_split
        self.data = encoded
        """

    def __getitem__(self, index):
        src, tgt = self.data[self.split][index]
        src, tgt = FloatTensor([src]), LongTensor(tgt)
        return src, tgt

    def __len__(self):
        return len(self.data[self.split])

    def set_split(self, s):
        self.split = s

    def _download_and_extract(self):
        raise NotImplementedError
