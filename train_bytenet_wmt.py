import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from data.wmt_loader import WMT
from bytenet.bytenet_modules import BytenetEncoder, BytenetDecoder
from bytenet.beam_opennmt import Beam
import json

use_cuda = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
print("Use CUDA on {} devices: {}".format(ngpu, use_cuda))

config = json.load(open("config.json"))

ds = WMT(config["WMT_DIR"])
dl = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

de_labeler = ds.labelers[ds.split][1]
de_rlabeler = list(de_labeler)

num_classes = len(de_labeler)
input_features = 50 # 800 in paper
max_r = 16
k_enc = k_dec = 3
num_sets = 3 # 6 in paper

lr = 0.0003 # from paper

epochs = 2

beam_size = 12 # from paper
pad = len(ds.labelers[ds.split][1])
eos = 0
n_best = 3

encoder = BytenetEncoder(input_features//2, max_r, k_enc, num_sets)
decoder = BytenetDecoder(input_features//2, max_r, k_dec, num_sets, num_classes)
beam = Beam(12, pad, eos, n_best)

if use_cuda:
    encoder = nn.DataParallel(encoder).cuda() if ngpu > 1 else encoder.cuda()
    decoder = nn.DataParallel(decoder).cuda() if ngpu > 1 else decoder.cuda()

params = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]
#print(decoder)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params, lr)

for epoch in range(epochs):
    print("Epoch {}".format(epoch+1))
    for i, (mb, tgts) in enumerate(dl):
        encoder.zero_grad()
        decoder.zero_grad()
        encoder.train()
        decoder.train()
        if use_cuda:
            mb, tgts = mb.cuda(), tgts.cuda()
        mb, tgts = Variable(mb), Variable(tgts)
        mb = encoder(mb)
        out = decoder(mb)
        loss = criterion(out.view(-1, num_classes), tgts.view(-1))
        print("loss: {}".format(loss.data[0]))
        loss.backward()
        optimizer.step()
