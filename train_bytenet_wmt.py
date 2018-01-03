import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from data.wmt_loader import WMT, PadCollate
from bytenet.bytenet_modules import BytenetEncoder, BytenetDecoder
from bytenet.beam_opennmt import Beam
import json

parser = argparse.ArgumentParser(description='PyTorch Bytenet WMT Trainer')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--d', type=int, default=400, metavar="d",
                    help='number of features in network (d)')
parser.add_argument('--max-r', type=int, default=16, metavar="r",
                    help='max dilation size (max r)')
parser.add_argument('--nsets', type=int, default=6,
                    help='number of ResBlock sets')
parser.add_argument('--k', type=int, default=3,
                    help='kernel size')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="bytenet_wmt",
                    help='model name')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
print("Use CUDA on {} devices: {}".format(ngpu, use_cuda))

config = json.load(open("config.json"))

ds = WMT(config["WMT_DIR"])
de_labeler = ds.labelers[ds.split][1]
de_rlabeler = list(de_labeler)

ignore_idx = -100
pad_vals = (len(de_labeler)-1, ignore_idx)
dl = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                     drop_last=True, num_workers=args.num_workers,
                     collate_fn=PadCollate(pad_vals=pad_vals))

num_classes = len(de_labeler)
input_features = args.d # 800 in paper
max_r = args.max_r
k_enc = k_dec = args.k
num_sets = args.nsets # 6 in paper

lr = args.lr # from paper

epochs = args.epochs

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

criterion = nn.NLLLoss(ignore_index=ignore_idx)
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
        loss = criterion(out.unsqueeze(2), tgts.unsqueeze(1)) # ach, alles für Bilder
        print("loss: {}".format(loss.data[0]))
        loss.backward()
        optimizer.step()
