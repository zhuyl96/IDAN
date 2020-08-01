# -*- coding: utf-8 -*-
"""
@Time : 2020/1/13 14:30 
@Author : zhuyl
@Email : ylzhu1996@qq.com
@File : nlpcc2014.py 
@Software: PyCharm
@Desc: 

"""
import pandas as pd
import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer
from config.config import *
from preprocessing.utils import load_user_lexicon_en, extract_senti_words_en,extract_embedding
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K

# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(data_path):
    with open(os.path.join(data_path, 'rt-polarity.pos'), "r", encoding='ISO-8859-1') as f:
        pos_reviews = list()
        for line in f:
            review = list()
            review.append(line.strip())
            review = clean_str(" ".join(review))
            pos_reviews.append(review)
        f.close()
    label = np.ones((len(pos_reviews),), dtype=int)  # 标签
    label = label.tolist()
    data_dict = {'label': label, 'review': pos_reviews}
    df_pos = pd.DataFrame(data_dict)

    with open(os.path.join(data_path, 'rt-polarity.neg'), "r", encoding='ISO-8859-1') as f:
        neg_reviews = list()
        for line in f:
            review = list()
            review.append(line.strip())
            review = clean_str(" ".join(review))
            neg_reviews.append(review)
        f.close()
    label = np.zeros((len(neg_reviews),), dtype=int)
    label = label.tolist()
    data_dict = {'label': label, 'review': neg_reviews}
    df_neg = pd.DataFrame(data_dict)

    pos_train_set, pos_test_set, neg_train_set, neg_test_set = train_test_split(df_pos, df_neg, random_state=42,
                                                                                test_size=0.2)  # 划分20%作为测试集

    df = pd.concat([pos_train_set, neg_train_set])
    df = df.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    df['review_'] = df['review'].map(lambda x: Lem(x))  # 词形还原

    df_test = pd.concat([pos_test_set, neg_test_set])
    df_test = df_test.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    df_test['review_'] = df_test['review'].map(lambda x: Lem(x))  # 词形还原

    return df, df_test


def Lem(text):
    words = text.split()
    wnl = WordNetLemmatizer()  # 词形还原
    for i in range(len(words)):
        words[i] = wnl.lemmatize(words[i], )  # 默认还原为名词
        words[i] = wnl.lemmatize(words[i], 'v')  # 还原为动词
    return " ".join(words)


if __name__ == '__main__':
    mr_args = mr_parser()
    utils_args = utils_parser()

    data_path = "./../data/raw data/MR"
    train_set, test_set = load_data(data_path)
    print(train_set.shape, test_set.shape)
    sentiment_words, adverb_words = load_user_lexicon_en(mr_args.lexicon_basepath, mr_args.lexicon_names)
    print(len(sentiment_words), len(adverb_words))
    # review_是经过词形还原的 仅用于提取sent words
    train_set['senti_words'] = train_set['review_'].map(
        lambda x: extract_senti_words_en(sentiment_words, adverb_words, str(x)))
    test_set['senti_words'] = test_set['review_'].map(
        lambda x: extract_senti_words_en(sentiment_words, adverb_words, str(x)))

    train_set['sum'] = train_set['review'].map(lambda x: len(str(x).split()))
    train_set['sum_senti_words'] = train_set['senti_words'].map(lambda x: len(x.split()))

    train_set.to_csv(os.path.join(data_path, 'train_set.csv'), index=False, encoding='utf-8-sig')
    test_set.to_csv(os.path.join(data_path, 'test_set.csv'), index=False, encoding='utf-8-sig')

    extract_embedding(model_path=utils_args.bert_path_en, data=train_set.review, seq_length=mr_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'MR/train.npy'))  # train review
    extract_embedding(model_path=utils_args.bert_path_en, data=train_set.senti_words, seq_length=mr_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'MR/senti_train.npy'))  # trian sentiment

    extract_embedding(model_path=utils_args.bert_path_en, data=test_set.review, seq_length=mr_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'MR/test.npy'))  # test review
    extract_embedding(model_path=utils_args.bert_path_en, data=test_set.senti_words, seq_length=mr_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'MR/senti_test.npy'))  # test sentiment
