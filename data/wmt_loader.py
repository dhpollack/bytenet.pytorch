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
    UNK = "☺" # this will be used as the pad/spacer token

    def __init__(self, root, transform=None, target_transform=None,
                 split="europarl", use_str=False, prepad=True,
                 a=1.2, b=0, download=False, keep_files=False, infer=False):
        self.root = root
        self.keep_files = keep_files
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.use_str = use_str
        self.prepad = prepad
        self.a = a - 1. # from paper a = 1.2
        self.b = b # from paper b = 0

        if infer:
            pass
        else:

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

                assert len(english) == len(deutsch)

                if self.prepad:
                    data[k] = [self.pad_src_tgt(e, d, (uniq_en, uniq_de)) \
                               for e, d in zip(english, deutsch) \
                               if len(e) != 0 \
                               and len(e) < 1000 and len(d) < 1000 \
                               and (len(d) / len(e) <= a*.98 and len(d) / len(e) > 0.3)]
                else:
                    data[k] = [(e, d) for e, d in zip(english, deutsch) \
                               if len(e) != 0 \
                               and len(e) < 1000 and len(d) < 1000 \
                               and (len(d) / len(e) <= a*.98 and len(d) / len(e) > 0.3)]
                uniq[k] = (uniq_en, uniq_de)
                #print(len(english), len(english) - len(data[k]))

            self.labelers = uniq
            self.data = data

    def __getitem__(self, index):
        src, tgt = self.data[self.split][index]
        if not self.prepad:
            src, tgt = self.pad_src_tgt(src, tgt, self.labelers[self.split])
        if not self.use_str:
            src = FloatTensor([src])
            tgt = LongTensor(tgt)
        return src, tgt

    def __len__(self):
        return len(self.data[self.split])

    def set_split(self, s):
        self.split = s

    def pad_src_tgt(self, src, tgt, labelers):
        src_labeler, tgt_labeler = labelers
        if self.use_str:
            # pad source
            src_pad = src + [list(src_labeler)[self.END]]
            src_pad += [self.UNK] * int(len(src) * self.a + self.b)
            # pad target
            tgt_pad = tgt + [list(tgt_labeler)[self.END]]
            if len(src_pad) > len(tgt_pad):
                tgt_pad += [self.UNK] * (len(src_pad) - len(tgt_pad))
        else:
            # pad source
            src_pad = [src_labeler[c_src] for c_src in src]
            src_pad += [self.END]
            src_pad += [len(src_labeler)-1] * int(len(src) * self.a + self.b)
            # pad target
            tgt_pad = [tgt_labeler[c_tgt] for c_tgt in tgt]
            tgt_pad += [self.END]
            if len(src_pad) > len(tgt_pad):
                tgt_pad += [len(tgt_labeler)-1] * (len(src_pad) - len(tgt_pad))

        return src_pad, tgt_pad

    def _download_and_extract(self):
        raise NotImplementedError
