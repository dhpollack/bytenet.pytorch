import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from data.enwik8_loader import WIKIPEDIA
from bytenet.bytenet_modules import BytenetDecoder, SimpleEmbEncoder
import json

use_cuda = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
print("Use CUDA on {} devices: {}".format(ngpu, use_cuda))

config = json.load(open("config.json"))

num_classes = 205
input_features = 50
max_r = 16
kernel_size = 3
num_sets = 6

encoder = SimpleEmbEncoder(num_classes, input_features)
decoder = BytenetDecoder(input_features//2, max_r, kernel_size, num_sets, num_classes)
if use_cuda:
    encoder = nn.DataParallel(encoder).cuda() if ngpu > 1 else encoder.cuda()
    decoder = nn.DataParallel(decoder).cuda() if ngpu > 1 else decoder.cuda()
#print(decoder)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD([{"params": encoder.parameters()}, {"params": decoder.parameters()}], 0.001, 0.9)

ds = WIKIPEDIA(config["HUTTER_DIR"])
dl = data.DataLoader(ds, batch_size=100, shuffle=False, num_workers=8)

for i, (mb, tgts) in enumerate(dl):
    #print(mb.size(), tgts.size())
    encoder.zero_grad()
    decoder.zero_grad()
    if use_cuda:
        mb, tgts = mb.cuda(), tgts.cuda()
    mb, tgts = Variable(mb), Variable(tgts)
    mb = encoder(mb)
    #print(mb.size())
    if ngpu > 1:
        out = decoder.module.generate(mb, 400, encoder)
    else:
        out = decoder.generate(mb, 400, encoder)
    loss = criterion(out.view(-1, num_classes), tgts.view(-1))
    print("loss: {}".format(loss.data[0]))
    loss.backward()
    optimizer.step()
