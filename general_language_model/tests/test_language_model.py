# encoding: utf-8
"""
    test_language_model.py
    ~~~~~~~~~
    
    Model description
    
    :author: Xiaofei Sun
    :contact: adoni1203@gmail.com
    :date: 2018/5/29
    :license: MIT, see LICENSE for more details.
"""

from general_language_model.language_model import LanguageModel
import numpy

lm = LanguageModel(None, None, './model')

while 1:
    sentence = input('Sentence: ')
    sentence = sentence.split() + ['<eos>']
    # score = lm.scoring(sentence)
    # scores = [score]
    # for i in range(10):
    #     numpy.random.shuffle(sentence)
    #     score = lm.scoring(sentence)
    #     scores.append(score)
    # print('Original Sentence')
    # print(scores[0])
    # print('Shuffled Sentences')
    # print(scores[1:])
    words, score = lm.predict_next_word(sentence)
    argsort = numpy.argsort(-1*score)
    for idx in argsort[:10]:
        print(" ".join(sentence[:-1]) + " " + words[idx])

