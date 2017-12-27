# Introduction
Bytenet (PyTorch)

This is a PyTorch version of [Bytenet](https://arxiv.org/abs/1610.10099), a fully convolutional encoder-decoder network for machine translation.  Notable features of this network include the stacked nature of the encoder and the decoder, the dilated (atrous) convolutions of exponentially expanding size in the encoder and decoder, and the masked (casual) convolutions in the decoder.

# Requirements
* Python 3.6 (may work with other versions, but I used 3.6)
* [PyTorch 0.3 or greater](http://pytorch.org/)
* [sacrebleu](https://github.com/awslabs/sockeye) - BLEU scoring

# Datasets
* [WMT News Translation '13, '14, and '15 de-en](http://www.statmt.org/wmt13/translation-task.html#download) - Translation task
* [Hutter Prize Wikipedia](http://prize.hutter1.net) - used to test decoder net only

# How to use
* First edit, config.json to point to the location of your datasets
