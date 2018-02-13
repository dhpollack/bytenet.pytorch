from __future__ import print_function
from torch import LongTensor, FloatTensor
import torch.utils.data as data
from torch.autograd import Variable
import os
import errno
import math
from collections import OrderedDict, Counter
from utils import *

class TABOETA(data.Dataset):
    """`Taboeta <http://tatoeba.org/eng/downloads>`_ Dataset.

    Args:
        TODO: update documentation
        root (string): Root directory of dataset.
        keep_files(bool, optional): if true, clean up is not performed on downloaded
            files.
    """

    SPLITS = {
        "en-de": ("deu.txt"),
    }

    REPLACE = [("—", "–"), ("―", "–"), ("−", "–"), (u'\u200b', ""), (u'\xa0', " ")]

    END = 0 # this is the "\n" character which gets removed with the split op
    UNK = "☺" # this will be used as the pad/spacer token

    def __init__(self, root, transform=None, target_transform=None,
                 split="en-de", use_str=False, prepad=True, calc_freq=False,
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
            if calc_freq: freq = {}
            for k, v in self.SPLITS.items():
                split_dir = os.path.join(root, k)
                with open(os.path.join(split_dir, v), "r") as f:
                    raw = f.read()
                    raw = self.replace_chars(raw)

                # find unique letters
                pairs = [l for l in raw.split("\n") if l.count("\t") > 0]
                pairs = [p.split("\t") for p in pairs]
                sentences_p1, sentences_p2 = zip(*pairs)
                chars_p1, chars_p2 = ''.join(sentences_p1), ''.join(sentences_p2)
                uniq_p1 = sorted(list(set(chars_p1)))
                uniq_p1 = ["\n"] + uniq_p1
                uniq_p1 += [self.UNK]
                #uniq_p1 = OrderedDict([(k, i) for i, k in enumerate(uniq_p1)])
                #uniq_p1.update({self.UNK: len(uniq_p1)})
                uniq_p2 = sorted(list(set(chars_p2)))
                uniq_p2 = ["\n"] + uniq_p2
                uniq_p2 += [self.UNK]
                #uniq_p2 = OrderedDict([(k, i) for i, k in enumerate(uniq_p2)])
                #uniq_p2.update({self.UNK: len(uniq_p2)})
                uniq[k] = (uniq_p1, uniq_p2)

                # character frequency count, not used yet
                if calc_freq:
                    freq_p1, freq_p2 = Counter(chars_p1), Counter(chars_p2)
                    freq_p1 = [freq_p1[k]/len(chars_p1) if k in freq_p1 else 1. for k in uniq_p1]
                    freq_p1[0] = len(sentences_p1) / len(chars_p1)
                    freq_p2 = [freq_p2[k]/len(chars_p2) if k in freq_p2 else 1. for k in uniq_p2]
                    freq_p2[0] = len(sentences_p2) / len(chars_p2)
                    freq[k] = (freq_p1, freq_p2)

                assert len(sentences_p1) == len(sentences_p2)


                if self.prepad:
                    data[k] = [self.pad_src_tgt(p1, p2, (uniq_p1, uniq_p2)) \
                               for p1, p2 in zip(sentences_p1, sentences_p2) \
                               if len(p1) < 1000 and len(p2) < 1000 \
                               and (len(p2) / len(p1) <= a*.98 and len(p2) / len(p1) > 0.3)]
                else:
                    data[k] = [(p1, p2) for p1, p2 in zip(sentences_p1, sentences_p2) \
                               if len(p1) < 1000 and len(p2) < 1000 \
                               and (len(p2) / len(p1) <= a*.98 and len(p2) / len(p1) > 0.3)]

            self.labelers = uniq
            self.data = data
            if calc_freq: self.charfreqs = freq

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
            src_pad = src + [src_labeler[self.END]]
            src_pad += [self.UNK] * int(len(src) * self.a + self.b)
            # pad target
            tgt_pad = tgt + [tgt_labeler[self.END]]
            if len(src_pad) > len(tgt_pad):
                tgt_pad += [self.UNK] * (len(src_pad) - len(tgt_pad))
        else:
            # pad source
            src_pad = [src_labeler.index(c_src) for c_src in src]
            src_pad += [self.END]
            src_pad += [len(src_labeler)-1] * int(len(src) * self.a + self.b)
            # pad target
            tgt_pad = [tgt_labeler.index(c_tgt) for c_tgt in tgt]
            tgt_pad += [self.END]
            if len(src_pad) > len(tgt_pad):
                tgt_pad += [len(tgt_labeler)-1] * (len(src_pad) - len(tgt_pad))

        return src_pad, tgt_pad

    def _download_and_extract(self):
        raise NotImplementedError

    def replace_chars(self, mystring, replace=None):
        if replace is None:
            replace = self.REPLACE
        for s, r in self.REPLACE:
            mystring = mystring.replace(s, r)
        return mystring
