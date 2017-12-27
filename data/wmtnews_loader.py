from __future__ import print_function
import torch.utils.data as data
from torch.autograd import Variable
import os
import errno
import math
from utils import *

class WMT(data.Dataset):
    """`WMT News Commentary <http://www.statmt.org/wmt13/>`_ Dataset.

    This loads the WMT news dataset for 2013, 2014, and 2015.

    Args:
        TODO: update documentation
        root (string): Root directory of dataset.
        keep_files(bool, optional): if true, clean up is not performed on downloaded
            files.
    """

    SPLITS = {
        "2013_news": ("news-commentary-v8.de-en.en", "news-commentary-v8.de-en.de"),
        "2014_news": ("news-commentary-v9.de-en.en", "news-commentary-v9.de-en.de"),
        "2015_news": ("news-commentary-v10.de-en.en", "news-commentary-v10.de-en.de"),
    }

    def __init__(self, root, transform=None, target_transform=None,
                 split="2013_news", use_cache=False, randomize=False,
                 download=False, keep_files=False):
        self.root = root
        self.keep_files = keep_files
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        data = {}
        uniq = set([])
        for k, v in self.SPLITS.items():
            split_dir = os.path.join(root, k)
            with open(os.path.join(split_dir, v[0]), "rb") as f:
                english = [[c for c in l] for l in f.readlines()]
                uniq.update([c for l in english for c in l])
            with open(os.path.join(split_dir, v[1]), "rb") as f:
                deutsch = [[c for c in l] for l in f.readlines()]
                uniq.update([c for l in deutsch for c in l])
            assert len(english) == len(deutsch)
            data[k] = [(e, d) for e, d in zip(english, deutsch) \
                       if (len(d) / len(e) < 1.5 and len(d) / len(e) > 0.3)]
            print(len(english), len(english) - len(data[k]))

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

    def __getitem__(self, index):
        src, tgt = self.data[self.split][index]
        return src, tgt

    def __len__(self):
        return len(self.data[self.split]) - sum(self.bs)

    def set_split(self, s):
        self.split = s

    def _download_and_extract(self):
        raise NotImplementedError
