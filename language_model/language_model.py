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


class LanguageModel(object):
    """
    General language model class
    """

    def __init__(self, model_path):
        pass

    def load(self, path):
        """

        :param path:
        :return:
        """

    def train(self):
        """

        :return:
        """

    def evaluate(self):
        """

        :return:
        """

    def predict_next_word(self, prefix):
        """
        :param prefix:
        :return: a word which is most possible to be the next word of this prefix
        """
        pass

    def predict_sentence(self, prefix):
        pass

    def predict_middle_word(self, prefix, postfix):
        pass

    def scoring(self, sentence):
        pass