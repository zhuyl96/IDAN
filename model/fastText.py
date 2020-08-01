# -*- coding: utf-8 -*-
"""
@Time : 20/7/11 20:31 
@Author : zhuyl
@File : fastText.py 
@Desc: 

"""
import fasttext
import pandas as pd
import numpy as np
import os
from config.config import *
import jieba


class fastText:
    def __init__(self, args):
        self.args = args

    def load_data(self, args):
        train_set = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'train_set.csv'), encoding='utf-8')
        test_set = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'test_set.csv'), encoding='utf-8')

        train_data = open(file=os.path.join(args.label_path, 'train_set.txt'), mode='w', encoding='utf-8')
        for i in train_set.itertuples():
            if 'cn' or 'chn' in args.label_path:
                review = ' '.join(jieba.cut(str(i.review)))
            else:
                review = i.review
            if i.label == 0:
                train_data.write('__label__negative ' + review + '\n')
            else:
                train_data.write('__label__positive ' + review + '\n')
        train_data.close()

        test_data = open(file=os.path.join(args.label_path, 'test_set.txt'), mode='w', encoding='utf-8')
        for i in test_set.itertuples():
            if 'cn' or 'chn' in args.label_path:
                review = ' '.join(jieba.cut(str(i.review)))
            else:
                review = i.review
            if i.label == 0:
                test_data.write('__label__negative ' + review + '\n')
            else:
                test_data.write('__label__positive ' + review + '\n')
        test_data.close()

    def model(self, args):
        self.load_data(args)
        model = fasttext.train_supervised(input=os.path.join(args.label_path, 'train_set.txt'), dim=300,
                                          lr=0.1, epoch=50, word_ngrams=2, loss='softmax',
                                          # autotuneValidationFile=os.path.join(args.label_path, 'test_set.txt'),
                                          # autotuneDuration=300,
                                          )

        def print_results(N, p, r):
            print("N\t" + str(N))
            print("P@{}\t{:.4f}".format(1, p))
            print("R@{}\t{:.4f}".format(1, r))

        print_results(*model.test(os.path.join(args.label_path, 'test_set.txt')))
        # print(model.test(os.path.join(args.label_path, 'test_set.txt')))


if __name__ == '__main__':
    args = nlpcc_parser()
    for i in range(1):
        print("----------------------------------------------------------------------------")
        print("----------------------------------", i + 1, "---------------------------------------")
        model = fastText(args=args, )
        model.model(args)
