# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 20:52 
@Author : zhuyl
@Email : ylzhu1996@qq.com
@File : IDAN.py
@Software: PyCharm
@Desc: 

"""

import numpy as np
import pandas as pd

import os
from config.config import *
from preprocessing.utils import w2v_prep_, ind2vec, w2v_prep_en_
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score


class SVM:
    def __init__(self, args, ):
        self.args = args
        self.seq_len = args.seq_len_w2v
        self.data_path = args.data_path_w2v
        self.word2vec_path = args.word2vec_path

    def load_data(self, args):
        word_dict, embedding_matrix, train, test = w2v_prep_en_(data_path=self.data_path,
                                                                w2v_path=self.word2vec_path,
                                                                seq_len=self.seq_len)
        train, test = ind2vec(embedding_matrix=embedding_matrix, train=train, test=test)

        train_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'train_set.csv'))
        train_label = np.asarray(list(train_label.label))
        test_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'test_set.csv'))
        test_label = np.asarray(list(test_label.label))
        print(train.shape, test.shape, train_label.shape, test_label.shape)

        return train, train_label, test, test_label

    def model(self, args):
        """

        :param args:
        :return:
        """
        train, train_label, test, test_label = self.load_data(args)

        clf = SVC(kernel='linear')
        clf.fit(X=train, y=train_label)
        y_pred = clf.predict(X=test)
        print("Accuracy: %0.4f." % accuracy_score(test_label, y_pred))
        print("F1-macro: %0.4f." % f1_score(test_label, y_pred, average='macro'))
        print('Reports\n', classification_report(test_label, y_pred))

        del clf


if __name__ == '__main__':
    args = nlpcc_en_parser()
    for i in range(12):
        print("----------------------------------------------------------------------------")
        print("----------------------------------", i + 1, "---------------------------------------")
        chn = SVM(args=args, )
        chn.model(args)
