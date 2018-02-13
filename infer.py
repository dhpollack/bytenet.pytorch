import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from data import select_dataset
from data.loader_utils import PadCollate, decode_one_sample
from bytenet.bytenet_modules import BytenetEncoder, BytenetDecoder
import json
import os

parser = argparse.ArgumentParser(description='PyTorch Bytenet Inference')
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')
parser.add_argument('--d', type=int, default=400, metavar="d",
                    help='number of features in network (d)')
parser.add_argument('--max-r', type=int, default=16, metavar="r",
                    help='max dilation size (max r)')
parser.add_argument('--nsets', type=int, default=6,
                    help='number of ResBlock sets')
parser.add_argument('--k', type=int, default=3,
                    help='kernel size')
parser.add_argument('--model-name', type=str, default="bytenet_wmt",
                    help='model name')
parser.add_argument('--load-model', type=str, default=None,
                    help='path of model to load')
parser.add_argument('--load-labelers', type=str, default=None,
                    help='path of labers json file')
parser.add_argument('--use-half-precision', action='store_true',
                    help='do all calculations in half precision')
parser.add_argument('mystring', metavar='s', type=str,
                    help='english string to be translated')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
ngpu = torch.cuda.device_count()
print("Use CUDA on {} devices: {}".format(ngpu, use_cuda))

config = json.load(open("config.json"))

if args.load_labelers is None or not os.path.exists(args.load_labelers):
    ds = select_dataset(args.model_name, config)
    src_labeler = ds.labelers[ds.split][0]
    tgt_labeler = ds.labelers[ds.split][1]
    if args.load_labelers is not None:
        json.dump((src_labeler, tgt_labeler), open(args.load_labelers, 'w'), sort_keys=True, indent=4)
else:
    ds = select_dataset(args.model_name, config, infer=True)
    src_labeler, tgt_labeler = json.load(open(args.load_labelers, 'r'))

src_rlabeler = list(src_labeler)
tgt_rlabeler = list(tgt_labeler)

print("src_labeler: {}, tgt_labeler: {}".format(len(src_labeler), len(tgt_labeler)))

ignore_idx = -100
pad_vals = (len(src_labeler)-1, ignore_idx)
num_classes = len(tgt_labeler)
input_features = args.d # 800 in paper
max_r = args.max_r
k_enc = k_dec = args.k
num_sets = args.nsets # 6 in paper

mystring = ds.replace_chars(args.mystring)

mystring_coded, _ = ds.pad_src_tgt(mystring, '', (src_labeler, tgt_labeler))

print("reverse src: {}".format(''.join([src_rlabeler[c] for c in mystring_coded])))

mystring_coded = torch.FloatTensor([mystring_coded])

mystring_coded.unsqueeze_(0)

if use_cuda:
    mystring_coded = mystring_coded.cuda()
if args.use_half_precision:
    mystring_coded = mystring_coded.half()

encoder = BytenetEncoder(input_features//2, max_r, k_enc, num_sets)
decoder = BytenetDecoder(input_features//2, max_r, k_dec, num_sets, num_classes, use_logsm=False)
#beam = Beam(12, pad, eos, n_best)

if use_cuda:
    encoder = nn.DataParallel(encoder).cuda() if ngpu > 1 else encoder.cuda()
    decoder = nn.DataParallel(decoder).cuda() if ngpu > 1 else decoder.cuda()

if args.load_model is not None:
    enstate, destate = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(enstate)
    decoder.load_state_dict(destate)

encoder.eval()
decoder.eval()

mb = Variable(mystring_coded, volatile=False)
mb = encoder(mb)
out = decoder(mb)

mystring_translated = decode_one_sample(out, tgt_rlabeler)

print(mystring_translated)
