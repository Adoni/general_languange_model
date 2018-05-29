# encoding: utf-8
"""
    train_model.py.py
    ~~~~~~~~~

    Train language model and save it to disk

    :author: Xiaofei Sun
    :contact: adoni1203@gmail.com
    :date: 2018/5/29
    :license: MIT, see LICENSE for more details.
"""

import torch
import argparse
from general_language_model.utils import Dictionary, batchify
import os
from general_language_model import RNNModel
from general_language_model.language_model import LanguageModel
import torch.nn as nn
import torch.onnx
import time
import math
from general_language_model.utils import get_batch, repackage_hidden


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--model-path', type=str, default='./model',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


dictionary = Dictionary()
train = dictionary.tokenize_path(os.path.join(args.data, 'train.txt'))
if os.path.exists(os.path.join(args.data, 'valid.txt')):
    valid = dictionary.tokenize_path(os.path.join(args.data, 'valid.txt'))
    exist_valid = True
else:
    exist_valid = False
if os.path.exists(os.path.join(args.data, 'test.txt')):
    test = dictionary.tokenize_path(os.path.join(args.data, 'test.txt'))
    exist_test = True
else:
    exist_test = False


eval_batch_size = 10
train_data = batchify(train, args.batch_size, device)
val_data = batchify(valid, eval_batch_size, device)
test_data = batchify(test, eval_batch_size, device)


ntokens = len(dictionary)

lm = LanguageModel(
    model=RNNModel.RNNModel(
        args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device),
    dictionary=dictionary,
    model_path=args.model_path)
lm.save()

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.


def train():
    # Turn on training mode which enables dropout.
    lm.model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(dictionary)
    hidden = lm.model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        lm.model.zero_grad()
        output, hidden = lm.model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(lm.model.parameters(), args.clip)
        for p in lm.model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        val_loss = lm.evaluate(val_data, eval_batch_size, args.bptt, criterion)
        train()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            lm.save()
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

lm.load()

# Run on test data.
test_loss = lm.evaluate(test_data, eval_batch_size, args.bptt, criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
