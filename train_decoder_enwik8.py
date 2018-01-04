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
input_features = 100 # 512 paper uses
max_r = 16
kernel_size = 3
num_sets = 6
n_samples = 400

epochs = 10

encoder = SimpleEmbEncoder(num_classes, input_features)
decoder = BytenetDecoder(input_features//2, max_r, kernel_size, num_sets, num_classes, use_logsm=False)
if use_cuda:
    encoder = nn.DataParallel(encoder).cuda() if ngpu > 1 else encoder.cuda()
    decoder = nn.DataParallel(decoder).cuda() if ngpu > 1 else decoder.cuda()
params = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params, 0.0003, weight_decay=0.0001)

ds = WIKIPEDIA(config["HUTTER_DIR"])
dl = data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

for epoch in range(epochs):
    print("Epoch {}".format(epoch+1))
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
            out = decoder.module.generate(mb, n_samples, encoder) # does not parallelize
        else:
            out = decoder.generate(mb, n_samples, encoder)
        loss = criterion(out.view(-1, num_classes), tgts.view(-1))
        print("loss: {} on e{}-b{}".format(loss.data[0], epoch+1, i+1))
        loss.backward()
        optimizer.step()
    mstate = (encoder.state_dict(), decoder.state_dict())
    sname = "output/states/{}_{}.pt".format("bytenet_decoder_enwik8", epoch+1)
    torch.save(mstate, sname)
