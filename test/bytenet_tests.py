import unittest
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from bytenet.bytenet_modules import *

class Test_ResBlock(unittest.TestCase):
    data = (torch.Tensor(1, 1000).uniform_() * 205).long()
    input_features = 50

    def test_1_resblock(self):
        mb = Variable(self.data)
        block = ResBlock(self.input_features // 2, 1, 3, True)
        encoder = nn.Embedding(num_embeddings=205, embedding_dim=self.input_features)
        mb_encoded = encoder(mb).transpose(1, 2)
        print(block(mb_encoded).size())

    def test_2_resblockset(self):
        mb = Variable(self.data)
        block = ResBlock(self.input_features // 2, 1, 3, True)
        encoder = nn.Embedding(num_embeddings=205, embedding_dim=self.input_features)
        mb_encoded = encoder(mb).transpose(1, 2)

        resbset = ResBlockSet(self.input_features // 2, 16, 3)
        print(resbset(mb_encoded).size())

    def test_3_bytenet_decoder(self):
        mb = Variable(self.data)
        encoder = nn.Embedding(num_embeddings=205, embedding_dim=self.input_features)
        mb_encoded = encoder(mb).transpose(1, 2)

        decoder = BytenetDecoder(self.input_features // 2, 16, 3, 6, 205)
        #print(decoder)
        out = decoder(mb_encoded)
        out = F.softmax(out, dim=1)
        print(out.size())

        decoder_red = BytenetDecoder(self.input_features // 2, 16, 3, 4, 205, [0, 0, 4, 4])
        #print(decoder_red)
        out_red = decoder_red(mb_encoded)
        out_red = F.softmax(out_red, dim=1)
        print(out_red.size())


    def test_4_bytenet_encoder(self):
        mb = Variable(self.data.float()).unsqueeze(1)
        encoder = BytenetEncoder(self.input_features // 2, 16, 3, 6)
        #print(encoder)
        mb_encoded = encoder(mb)
        print(mb_encoded)
        #decoder = BytenetDecoder(self.input_features // 2, 16, 3, 6, 205)
        #print(decoder)
        #out = decoder(mb_encoded)
        #out = F.softmax(out, dim=1)

        #print(out.size())

if __name__ == '__main__':
    unittest.main()
