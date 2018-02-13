import argparse
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

parser = argparse.ArgumentParser(description='Bytenet Hunter Challenge Decoder Trainer')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=4, metavar='bs',
                    help='batch size')
parser.add_argument('--input-features', type=int, default=512,
                    help='number of input features')
parser.add_argument('--num-classes', type=int, default=205,
                    help='number of output classes')
parser.add_argument('--max-r', type=int, default=16,
                    help='max dilation size')
parser.add_argument('--k', type=int, default=3,
                    help='kernel size')
parser.add_argument('--num-sets', type=int, default=6,
                    help='number of output classes')
parser.add_argument('--num-samples', type=int, default=400,
                    help='number of samples to create')
parser.add_argument('--data-path', type=str, default="data/voxforge",
                    help='data path')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers for data loader')
parser.add_argument('--validate', action='store_true',
                    help='do out-of-bag validation')
parser.add_argument('--log-interval', type=int, default=5,
                    help='reports per epoch')
parser.add_argument('--chkpt-interval', type=int, default=10,
                    help='how often to save checkpoints')
parser.add_argument('--model-name', type=str, default="bytenet_decoder_enwik8",
                    help='name of model')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--save-model', action='store_true',
                    help='path to save the final model')
args = parser.parse_args()

num_classes = args.num_classes
input_features = args.input_features # 512 paper uses
max_r = args.max_r
kernel_size = args.k
num_sets = args.num_sets
n_samples = args.num_samples

epochs = args.epochs

encoder = SimpleEmbEncoder(num_classes, input_features)
decoder = BytenetDecoder(input_features//2, max_r, kernel_size, num_sets, num_classes, use_logsm=False)
if use_cuda:
    encoder = nn.DataParallel(encoder).cuda() if ngpu > 1 else encoder.cuda()
    decoder = nn.DataParallel(decoder).cuda() if ngpu > 1 else decoder.cuda()
params = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]

criterion = nn.CrossEntropyLoss(size_average=True)
optimizer = torch.optim.Adam(params, 0.0003, weight_decay=0.0001)

ds = WIKIPEDIA(config["HUTTER_DIR"])
dl = data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

def train(epoch):
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
        print(out.size(), tgts.size())
        loss = criterion(out, tgts)
        print("loss: {} on e{}-b{}".format(loss.data[0], epoch+1, i+1))
        loss.backward()
        optimizer.step()
    mstate = (encoder.state_dict(), decoder.state_dict())
    sname = "output/states/{}_{}.pt".format(args.model_name, epoch+1)
    torch.save(mstate, sname)


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
