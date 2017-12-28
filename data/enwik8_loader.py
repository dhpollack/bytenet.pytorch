from __future__ import print_function
import torch
import torch.utils.data as data
from torch.autograd import Variable
import os
import errno
import math
from utils import *

class WIKIPEDIA(data.Dataset):
    """`Hunter Prize Wikipedia <http://prize.hunter1.net>`_ Dataset.

    This loads the Hunter Prize Wikipedia dataset as used in the Bytenet paper.

    Args:
        root (string): Root directory of dataset.
        bs (tuple[int]): (source, target) batch sizes
        download (bool, opt): if true, download dataset from internet
        keep_files (bool, opt): if true, clean up is not performed on downloaded
            files.
    """

    SPLITS = [("train", 90000000), ("valid", 95000000), ("test", 100000000)]

    def __init__(self, root, transform=None, target_transform=None,
                 split="train", bs=(100, 400), use_cache=False,
                 download=False, keep_files=False):
        self.root = root
        self.keep_files = keep_files
        self.split = split
        self.bs = bs
        self.transform = transform
        self.target_transform = target_transform

        enwik8_fp = os.path.join(root, "enwik8")

        if download:
            _make_dir_iff(self.root)
            self._download_and_extract()
        else:
            if not os.path.exists(enwik8_fp):
                raise FileExistsError("{} does not exist, use download=True".format(enwik8_fp))

        with open(enwik8_fp, "r") as f:
            raw = f.read()

        raw = [x for x in raw.encode("utf8")]
        uniq = sorted(list(set(raw)))
        labeler = {k: i for i, k in enumerate(uniq)}
        rlabeler = {v: k for k, v in labeler.items()}

        encoded = [labeler[x] for x in raw]

        data = {}
        for i, (k, v) in enumerate(self.SPLITS):
            if i == 0:
                data[k] = encoded[:v]
            else:
                data[k] = encoded[prev_v:v]
            prev_v = v

        self.data = data
        self.labeler = labeler
        self.rlabeler = rlabeler

    def __getitem__(self, index):
        st_src = index
        end_src = index + self.bs[0]
        st_tgt = end_src
        end_tgt = st_tgt + self.bs[1]
        src = self.data[self.split][st_src:end_src]
        tgt = self.data[self.split][st_tgt:end_tgt]
        src, tgt = torch.LongTensor(src), torch.LongTensor(tgt)
        return src, tgt

    def __len__(self):
        return len(self.data[self.split]) - sum(self.bs)

    def set_split(self, s):
        self.split = s

    def _download_and_extract(self):
        raise NotImplementedError
