# -*- coding: utf-8 -*-
"""
@Time : 2020/1/13 14:30 
@Author : zhuyl
@File : nlpcc2014.py
@Desc: 

"""
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
from config.config import nlpcc_parser, utils_parser
from preprocessing.utils import load_user_lexicon, extract_senti_words, extract_embedding
import tensorflow as tf
from keras import backend as K

# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


def load_data(data_path):
    with open(os.path.join(data_path, 'sample.positive.txt'), "r", encoding='utf-8') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        pos_reviews = reviews.find_all(name='review')
        for ind, i in enumerate(pos_reviews):
            pos_reviews[ind] = str(i.text).replace('\n', '')
    label = np.ones((len(pos_reviews),), dtype=int)
    label = label.tolist()
    data_dict = {'label': label, 'review': pos_reviews}
    df_pos = pd.DataFrame(data_dict)

    with open(os.path.join(data_path, 'sample.negative.txt'), "r", encoding='utf-8') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        neg_reviews = reviews.find_all(name='review')
        for ind, i in enumerate(neg_reviews):
            neg_reviews[ind] = str(i.text).replace('\n', '')
    label = np.zeros((len(neg_reviews),), dtype=int)
    label = label.tolist()
    data_dict = {'label': label, 'review': neg_reviews}
    df_neg = pd.DataFrame(data_dict)

    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    df['sum'] = df['review'].map(lambda x: len(str(x)))

    with open(os.path.join(data_path, 'test.label.cn.txt'), "r", encoding='utf-8') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        test_reviews = reviews.find_all(name='review')
        label = []
        for ind, i in enumerate(test_reviews):
            label.append(i.attrs['label'])
            test_reviews[ind] = str(i.text).replace('\n', '')
    data_dict = {'label': label, 'review': test_reviews}
    df_test = pd.DataFrame(data_dict)

    return df, df_test


if __name__ == '__main__':
    nlpcc_args = nlpcc_parser()
    utils_args = utils_parser()

    train_set, test_set = load_data(nlpcc_args.data_path)
    print(train_set.shape, test_set.shape)
    sentiment_words, adverb_words, negative_words = load_user_lexicon(utils_args.lexicon_basepath,
                                                                      utils_args.lexicon_names)
    print(len(sentiment_words), len(adverb_words), len(negative_words))

    train_set['senti_words'] = train_set['review'].map(
        lambda x: extract_senti_words(sentiment_words, adverb_words, negative_words, str(x)))
    test_set['senti_words'] = test_set['review'].map(
        lambda x: extract_senti_words(sentiment_words, adverb_words, negative_words, str(x)))

    train_set['sum_senti_words'] = train_set['senti_words'].map(lambda x: len(x))

    train_set.to_csv(os.path.join(nlpcc_args.data_path, 'train_set.csv'), index=False, encoding='utf-8-sig')
    test_set.to_csv(os.path.join(nlpcc_args.data_path, 'test_set.csv'), index=False, encoding='utf-8-sig')

    extract_embedding(model_path=utils_args.bert_path_cn, data=train_set.review, seq_length=nlpcc_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/cn/train.npy'))  # train review
    extract_embedding(model_path=utils_args.bert_path_cn, data=train_set.senti_words, seq_length=nlpcc_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/cn/senti_train.npy'))  # trian sentiment

    extract_embedding(model_path=utils_args.bert_path_cn, data=test_set.review, seq_length=nlpcc_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/cn/test.npy'))  # test review
    extract_embedding(model_path=utils_args.bert_path_cn, data=test_set.senti_words, seq_length=nlpcc_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/cn/senti_test.npy'))  # test sentiment
