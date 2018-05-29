# encoding: utf-8
"""
    utils.py
    ~~~~~~~~~
    
    Here are some helper classes and functions
    
    :author: Xiaofei Sun
    :contact: adoni1203@gmail.com
    :date: 2018/5/28
    :license: MIT, see LICENSE for more details.
"""
import os
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.frozen = False

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def freeze(self):
        self.frozen = True

    def __len__(self):
        return len(self.idx2word)

    def tokenize_sentence(self, words):
        """Tokenizes a text file."""
        # Add words to the dictionary
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            try:
                ids[i] = self.word2idx[word]
            except:
                ids[i] = self.word2idx["<unk>"]
        return ids

    def tokenize_path(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if not self.frozen:
                    for word in words:
                        self.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    try:
                        ids[token] = self.word2idx[word]
                    except:
                        ids[token] = self.word2idx["<unk>"]
                    token += 1
        return ids


def batchify(data, bsz, device):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    :param data:
    :param bsz:
    :return:
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

