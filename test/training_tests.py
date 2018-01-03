import unittest
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from bytenet.bytenet_modules import *
from data.wmt_loader import *

class Test_Training(unittest.TestCase):
    config = json.load(open("config.json"))
    bs = 1
    use_cuda = torch.cuda.is_available()
    input_features = 50
    epochs = 100

    def test_1_training(self):
        """
            This is a test to see if the encoder decoder network can memorize a
            single src-tgt pairing. The learning rate is intentionally high
            because we are doing memorization and not learning.
        """
        ds = WMT(self.config["WMT_DIR"])
        dl = data.DataLoader(ds, batch_size=self.bs)

        de_labeler = ds.labelers[ds.split][1]
        de_rlabeler = list(de_labeler)
        num_letters = len(de_labeler) # get the german labeler

        encoder = BytenetEncoder(self.input_features // 2, 16, 3, 2)
        decoder = BytenetDecoder(self.input_features // 2, 16, 3, 2, num_letters)
        params = [{"params": encoder.parameters()},
                  {"params": decoder.parameters()}]
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(params, lr=0.003)
        for src, tgt in dl:
            print(src.size(), tgt.size())
            print(tgt)
            if self.use_cuda:
                src, tgt = src.cuda(), tgt.cuda()
            src, tgt = Variable(src), Variable(tgt)
            break
        for epoch in range(self.epochs):
            encoder.zero_grad()
            decoder.zero_grad()
            encoder.train()
            decoder.train()
            out = encoder(src)
            out = decoder(out)
            loss = criterion(out.unsqueeze(2), tgt.unsqueeze(1))
            print(loss.data[0])
            loss.backward()
            optimizer.step()

        print(tgt.size())
        print(out.size(), out.data.max(1)[1].size())
        print("".join([de_rlabeler[c] for c in tgt.data.view(-1)]))
        print("".join([de_rlabeler[c] for c in out.data.max(1)[1].view(-1)]))

if __name__ == '__main__':
    unittest.main()
