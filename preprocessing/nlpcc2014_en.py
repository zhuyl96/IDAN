# -*- coding: utf-8 -*-
"""
@Time : 2020/1/13 14:30 
@Author : zhuyl
@File : nlpcc2014_en.py
@Desc: 

"""
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer
from config.config import nlpcc_en_parser, utils_parser
from preprocessing.utils import load_user_lexicon_en, extract_senti_words_en,extract_embedding
import tensorflow as tf
from keras import backend as K

# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

def load_data(data_path):
    results = re.compile(r'https?://[a-zA-Z0-9.?/&=:]*', re.S)  # 正则表达式去除文本中的URL
    with open(os.path.join(data_path, 'sample.positive.txt'), "r", encoding='utf-8') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        pos_reviews = reviews.find_all(name='review')
        for ind, i in enumerate(pos_reviews):
            i = results.sub("", i.text)  # 去除URL
            pos_reviews[ind] = str(i).replace('\n', '')
    label = np.ones((len(pos_reviews),), dtype=int)
    label = label.tolist()
    data_dict = {'label': label, 'review': pos_reviews}
    df_pos = pd.DataFrame(data_dict)

    with open(os.path.join(data_path, 'sample.negative.txt'), "r", encoding='utf-8') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        neg_reviews = reviews.find_all(name='review')
        for ind, i in enumerate(neg_reviews):
            i = results.sub("", i.text)
            neg_reviews[ind] = str(i).replace('\n', '')
    label = np.zeros((len(neg_reviews),), dtype=int)
    label = label.tolist()
    data_dict = {'label': label, 'review': neg_reviews}
    df_neg = pd.DataFrame(data_dict)

    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    df['review'] = df['review'].map(lambda x: str(x).lower())
    df['review_'] = df['review'].map(lambda x: Lem(x))  # 词形还原
    df['sum'] = df['review'].map(lambda x: len(str(x).split()))

    with open(os.path.join(data_path, 'test.label.en.txt'), "r", encoding='ISO-8859-1') as f:
        reviews = f.read()
        reviews = BeautifulSoup(reviews, 'html.parser')
        test_reviews = reviews.find_all(name='review')
        label = []
        for ind, i in enumerate(test_reviews):
            label.append(i.attrs['label'])
            i = results.sub("", i.text)
            test_reviews[ind] = str(i).replace('\n', '')
    data_dict = {'label': label, 'review': test_reviews}
    df_test = pd.DataFrame(data_dict)
    df_test['review'] = df_test['review'].map(lambda x: str(x).lower())
    df_test['review_'] = df_test['review'].map(lambda x: Lem(x))  # 词形还原

    return df, df_test


# 转换为小写 并 词形还原
def Lem(text):
    words = text.split()
    wnl = WordNetLemmatizer()  # 词形还原
    for i in range(len(words)):
        words[i] = wnl.lemmatize(words[i], )  # 默认还原为名词
        words[i] = wnl.lemmatize(words[i], 'v')  # 还原为动词
    return " ".join(words)


if __name__ == '__main__':
    nlpcc_args = nlpcc_en_parser()
    utils_args = utils_parser()

    data_path = "./../data/raw data/NLPCC 2014 Task2/en/"
    train_set, test_set = load_data(data_path)
    print(train_set.shape, test_set.shape)
    sentiment_words, adverb_words = load_user_lexicon_en(nlpcc_args.lexicon_basepath, nlpcc_args.lexicon_names)
    print(len(sentiment_words), len(adverb_words))

    train_set['senti_words'] = train_set['review_'].map(
        lambda x: extract_senti_words_en(sentiment_words, adverb_words, str(x)))
    test_set['senti_words'] = test_set['review_'].map(
        lambda x: extract_senti_words_en(sentiment_words, adverb_words, str(x)))

    train_set['sum_senti_words'] = train_set['senti_words'].map(lambda x: len(x.split()))

    train_set.to_csv(os.path.join(data_path, 'train_set.csv'), index=False, encoding='utf-8-sig')
    test_set.to_csv(os.path.join(data_path, 'test_set.csv'), index=False, encoding='utf-8-sig')

    extract_embedding(model_path=utils_args.bert_path_en, data=train_set.review, seq_length=nlpcc_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/en/train.npy'))  # train review
    extract_embedding(model_path=utils_args.bert_path_en, data=train_set.senti_words, seq_length=nlpcc_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/en/senti_train.npy'))  # trian sentiment

    extract_embedding(model_path=utils_args.bert_path_en, data=test_set.review, seq_length=nlpcc_args.seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/en/test.npy'))  # test review
    extract_embedding(model_path=utils_args.bert_path_en, data=test_set.senti_words, seq_length=nlpcc_args.senti_seq_len,
                      save_path=os.path.join(utils_args.save_path, 'NLPCC 2014 Task2/en/senti_test.npy'))  # test sentiment

