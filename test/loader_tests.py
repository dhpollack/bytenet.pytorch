import unittest
import json
import time
from data.enwik8_loader import *
from data.wmt_loader import *

class Loaders_Test(unittest.TestCase):
    config = json.load(open("config.json"))
    print(config)

    def test_1_enwik8(self):
        ds = WIKIPEDIA(self.config["HUTTER_DIR"])
        for i, (src, tgt) in enumerate(ds):
            print(len(src), len(tgt))
            print(src[:20], tgt[:20])
            if i > 0:
                break

    def test_2_wmtnews(self):
        ds = WMT(self.config["WMT_DIR"])
        for i, (src, tgt) in enumerate(ds):
            print(len(src), len(tgt))
            print(src, tgt)
            if i > 0:
                break

if __name__ == '__main__':
    unittest.main()
