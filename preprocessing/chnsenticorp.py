# -*- coding: utf-8 -*-
"""
@Time : 2020/1/1 14:38 
@Author : zhuyl
@File : IDAN.py
@Software: PyCharm
@Desc: 

"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from config.config import chn_parser, utils_parser
from preprocessing.utils import load_user_lexicon, extract_senti_words, extract_embedding
import tensorflow as tf
from keras import backend as K

# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


def load_data(data_base_path, subpath):
    """
    读取所有txt文件，并组成一个dataframe
    :param data_base_path: 数据所在的base path
    :param subpath: subpath
    :return: dataframe
    """
    # subpath='pos/pos.' or 'neg/neg.'
    reviews = labels = []
    for i in range(3000):
        with open(os.path.join(data_base_path, subpath + str(i) + '.txt'), "r", encoding='utf-8') as f:
            review = f.read()
            review = review.replace("\n", "")
            reviews.append(review)

    if subpath == 'pos/pos.':
        labels = np.ones((len(reviews),), dtype=int)
    elif subpath == 'neg/neg.':
        labels = np.zeros((len(reviews),), dtype=int)

    labels = labels.tolist()
    data_dict = {'label': labels, 'review': reviews}
    df = pd.DataFrame(data_dict)

    return df


def prepare_data(df_pos, df_neg):
    """
    在组合两种极性评论之前先将他们划分为训练集和测试集
    :param df_pos: positive reviews
    :param df_neg: negative reviews
    :return: train set, test set
    """
    pos_train_set, pos_test_set, neg_train_set, neg_test_set = train_test_split(df_pos, df_neg, random_state=42,
                                                                                test_size=0.2)
    train_set = pd.concat([pos_train_set, neg_train_set])
    test_set = pd.concat([pos_test_set, neg_test_set])
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    train_set['sum'] = train_set['review'].map(lambda x: len(str(x)))
    test_set['sum'] = test_set['review'].map(lambda x: len(str(x)))
    return train_set, test_set


if __name__ == '__main__':
    chn_args = chn_parser()
    utils_args = utils_parser()

    df_pos = load_data(chn_args.data_basepath, subpath='pos/pos.')
    df_neg = load_data(chn_args.data_basepath, subpath='neg/neg.')

    sentiment_words, adverb_words, negative_words = load_user_lexicon(utils_args.lexicon_basepath,
                                                                      utils_args.lexicon_names)
    print(len(sentiment_words), len(adverb_words), len(negative_words))

    train_set, test_set = prepare_data(df_pos, df_neg)
    print(train_set.shape, test_set.shape)
    train_set['senti_words'] = train_set['review'].map(
        lambda x: extract_senti_words(sentiment_words, adverb_words, negative_words, str(x)))
    test_set['senti_words'] = test_set['review'].map(
        lambda x: extract_senti_words(sentiment_words, adverb_words, negative_words, str(x)))

    train_set['sum_senti_words'] = train_set['senti_words'].map(lambda x: len(x))
    test_set['sum_senti_words'] = test_set['senti_words'].map(lambda x: len(x))

    train_set.to_csv(os.path.join(chn_args.data_basepath, 'train_set.csv'), index=False, encoding='utf-8-sig')
    test_set.to_csv(os.path.join(chn_args.data_basepath, 'test_set.csv'), index=False, encoding='utf-8-sig')

    extract_embedding(model_path=utils_args.bert_path_cn, data=train_set.review, seq_length=chn_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'ChnSentiCorp/train.npy'))  # train review
    extract_embedding(model_path=utils_args.bert_path_cn, data=train_set.senti_words, seq_length=chn_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'ChnSentiCorp/senti_train.npy'))  # trian sentiment

    extract_embedding(model_path=utils_args.bert_path_cn, data=test_set.review, seq_length=chn_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'ChnSentiCorp/test.npy'))  # test review
    extract_embedding(model_path=utils_args.bert_path_cn, data=test_set.senti_words, seq_length=chn_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'ChnSentiCorp/senti_test.npy'))  # test sentiment
