import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

def _same_pad(k=1, dil=1):
    # assumes stride length of 1
    # p = math.ceil((l - 1) * s - l + dil*(k - 1) + 1)
    p = math.ceil(dil*(k - 1))
    return p

class ResBlock(nn.Module):
    """
        Note To Self:  using padding to "mask" the convolution is equivalent to
        either centering the convolution (no mask) or skewing the convolution to
        the left (mask).  Either way, we should end up with n timesteps.

        Also note that "masked convolution" and "casual convolution" are two
        names for the same thing.

    Args:
        d (int): size of inner track of network.
        r (int): size of dilation
        k (int): size of kernel in dilated convolution (odd numbers only)
        casual (bool): determines how to pad the casual conv layer. See notes.
    """
    def __init__(self, d, r=1, k=3, casual=False, use_bias=False):
        super(ResBlock, self).__init__()
        self.d = d # input features
        self.r = r # dilation size
        self.k = k # "masked kernel size"
        ub = use_bias
        self.layernorm1 = nn.InstanceNorm1d(num_features=2*d, affine=True) # same as LayerNorm
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1x1_1 = nn.Conv1d(2*d, d, kernel_size=1, bias=ub) # output is "d"
        self.layernorm2 = nn.InstanceNorm1d(num_features=d, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        if casual:
            padding = (_same_pad(k,r) // 2, _same_pad(k,r) // 2)
        else:
            padding = (_same_pad(k,r), 0)
        self.pad = nn.ConstantPad1d(padding, 0.)
        #self.pad = nn.ReflectionPad1d(padding) # this might be better for audio
        self.maskedconv1xk = nn.Conv1d(d, d, kernel_size=k, dilation=r, bias=ub)
        self.layernorm3 = nn.InstanceNorm1d(num_features=d, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv1d(d, 2*d, kernel_size=1, bias=ub) # output is "d"

    def forward(self, input):
        x = input
        x = self.layernorm1(x)
        x = self.relu1(x)
        x = self.conv1x1_1(x)
        x = self.layernorm2(x)
        x = self.relu2(x)
        x = self.pad(x)
        x = self.maskedconv1xk(x)
        x = self.layernorm3(x)
        x = self.relu3(x)
        x = self.conv1x1_2(x)
        #print("ResBlock:", x.size(), input.size())
        x += input # add back in residual
        return x

class ResBlockSet(nn.Module):
    def __init__(self, d, max_r=16, k=3, casual=False):
        super(ResBlockSet, self).__init__()
        self.d = d
        self.max_r = max_r
        self.k = k
        rlist = [1 << x for x in range(15) if (1 << x) <= max_r]
        self.blocks = nn.Sequential(*[ResBlock(d, r, k, casual) for r in rlist])

    def forward(self, input):
        x = input
        x = self.blocks(x)
        return x

class BytenetEncoder(nn.Module):
    """
        d = hidden units
        max_r = maximum dilation rate (paper default: 16)
        k = masked kernel size (paper default: 3)
        num_sets = number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
        a = relative length of output sequence
        b = output sequence length intercept
    """
    def __init__(self, d=512, max_r=16, k=3, num_sets=6):
        super(BytenetEncoder, self).__init__()
        self.d = d
        self.max_r = max_r
        self.k = k
        self.num_sets = num_sets
        self.a = a
        self.b = b
        self.pad_in = nn.ConstantPad1d((0, 1), 0.)
        self.conv_in = nn.Conv1d(1, 2*d, 1)
        self.sets = nn.Sequential()
        for i in range(num_sets):
            self.sets.add_module("set_{}".format(i+1), ResBlockSet(d, max_r, k))
        self.conv_out = nn.Conv1d(2*d, 2*d, 1)

    def forward(self, input):
        x = input
        x_len = x.size(-1)
        x = F.pad(x, (0, math.ceil(x_len*(self.a-1.) + self.b)), mode='constant', value=0)
        x = self.conv_in(x)
        self.sets(x)
        x = self.conv_out(x)
        x = F.relu(x)
        return x

class SimpleEmbEncoder(nn.Module):
    def __init__(self, ne, ed):
        super(SimpleEmbEncoder, self).__init__()
        self.emb = nn.Embedding(num_embeddings=ne, embedding_dim=ed)
    def forward(self, input):
        x = self.emb(input)
        x = x.transpose(1, 2)
        return x

class BytenetDecoder(nn.Module):
    """
        d = hidden units
        max_r = maximum dilation rate (paper default: 16)
        k = masked kernel size (paper default: 3)
        num_sets = number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
        num_classes = number of output classes (Hunter prize default: 205)
    """
    def __init__(self, d=512, max_r=16, k=3, num_sets=6, num_classes=205,
                 reduce_out=None):
        super(BytenetDecoder, self).__init__()
        self.max_r = max_r
        self.k = k
        self.d = d
        self.num_sets = num_sets
        self.sets = nn.Sequential()
        for i in range(num_sets):
            self.sets.add_module("set_{}".format(i+1), ResBlockSet(d, max_r, k, True))
            if reduce_out is not None:
                r = reduce_out[i]
                if r != 0:
                    reduce_conv = nn.Conv1d(2*d, 2*d, r, r)
                    reduce_pad = nn.ConstantPad1d((0, r), 0.)
                    self.sets.add_module("reduce_pad_{}".format(i+1), reduce_pad)
                    self.sets.add_module("reduce_{}".format(i+1), reduce_conv)
        self.conv1 = nn.Conv1d(2*d, 2*d, 1)
        self.conv2 = nn.Conv1d(2*d, num_classes, 1)

    def forward(self, input):
        x = input
        x = self.sets(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = F.softmax(x, dim=2)
        x = x.transpose(1, 2).contiguous()
        return x

    def generate(self, input, n_samples, encoder=None):
        bs = input.size(0)
        x = input
        for i in range(n_samples):
            out = self(x)
            if i+1 != n_samples:
                gen = out.max(2)[1][:, -1].contiguous()
                gen = gen.view(bs, 1)
                gen_enc = encoder(gen)
                x = torch.cat((x, gen_enc), dim=2)
        # add last generated output to out
        gen = out[:, -1, :].unsqueeze(1)
        out = torch.cat((out, gen), dim=1)
        # return only generated outputs
        out = out[:,-n_samples:, :].contiguous()
        return out
