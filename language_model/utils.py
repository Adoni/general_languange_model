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

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def tokenize(path, dictionary):
    """Tokenizes a text file."""
    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, 'r') as f:
        tokens = 0
        for line in f:
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = dictionary.word2idx[word]
                token += 1

    return ids