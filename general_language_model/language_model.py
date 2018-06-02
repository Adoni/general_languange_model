# encoding: utf-8
"""
    language_model.py
    ~~~~~~~~~
    
    Model description
    
    :author: Xiaofei Sun
    :contact: adoni1203@gmail.com
    :date: 2018/5/28
    :license: MIT, see LICENSE for more details.
"""

from general_language_model.utils import Dictionary
import os
import torch
import pickle
from general_language_model.utils import get_batch
from general_language_model.utils import repackage_hidden
from general_language_model.utils import batchify
import torch.nn as nn


class LanguageModel(object):
    """
    General language model class
    """

    def __init__(self, model, dictionary, model_path):
        self.dictionary = dictionary
        self.model = model
        self.model_path = model_path
        if model is None:
            self.load()

    def load(self):
        """

        :param path:
        :return:
        """
        print('Loading from %s' % self.model_path)
        if not os.path.exists(os.path.join(self.model_path, 'model.pt')):
            print('No model file --> Stop loading')
            return
        with open(os.path.join(self.model_path, 'model.pt'), 'rb') as f:
            self.model = torch.load(f)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            self.model.rnn.flatten_parameters()

        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'rb') as f:
            self.dictionary = pickle.load(f)
            # after load the rnn params are not a continuous chunk of memory
        print('Finish loading')

    def train(self):
        """

        :return:
        """

    def evaluate(self, data_source, eval_batch_size, seq_len, criterion):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        ntokens = len(self.dictionary)
        hidden = self.model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, seq_len):
                data, targets = get_batch(data_source, i, seq_len)
                output, hidden = self.model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)

    def predict_next_word(self, prefix):
        """
        :param prefix:
        :return: a word which is most possible to be the next word of this prefix
        """
        prefix = self.dictionary.tokenize_sentence(prefix)
        prefix = prefix[:-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_batch_size, seq_len = 1, prefix.size(0)
        prefix = batchify(prefix, eval_batch_size, device)
        ntokens = len(self.dictionary)
        hidden = self.model.init_hidden(eval_batch_size)
        output, hidden = self.model(prefix, hidden)
        output_flat = output.view(-1, ntokens)
        last_one = output_flat[-1, :]
        return self.dictionary.idx2word, last_one.cpu().detach().numpy()

    def predict_sentence(self, prefix):
        pass

    def predict_middle_word(self, prefix, postfix):
        pass

    def scoring(self, sentence):
        sentence = self.dictionary.tokenize_sentence(sentence)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_batch_size, seq_len = 1, sentence.size(0)
        print(eval_batch_size, seq_len)
        sentence = batchify(sentence, eval_batch_size, device)
        criterion = nn.CrossEntropyLoss()
        score = self.evaluate(sentence, eval_batch_size, seq_len, criterion)
        return score

    def save(self):
        print('Save model and dictionary to %s' % self.model_path)
        with open(os.path.join(self.model_path, 'model.pt'), 'wb') as f:
            torch.save(self.model, f)
        with open(os.path.join(self.model_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self.dictionary, f)
