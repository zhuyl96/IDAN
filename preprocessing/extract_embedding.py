# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 16:56 
@Author : zhuyl
@Email : ylzhu1996@qq.com
@File : extract_embedding.py 
@Software: PyCharm
@Desc: 

"""

import pandas as pd
import numpy as np
import os
from config.config import *
from preprocessing.utils import extract_embedding, w2v_prep
import tensorflow as tf
from keras import backend as K

# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
# chnsenticorp
'''
data = pd.read_csv(os.path.join('./../data/raw data/ChnSentiCorp', 'ChnSentiCorp_test_set.csv'))  # 数据不同时要变
extract_embedding(model_path=args.roberta_path,  # 预训练语言模型地址
                  data=data.senti_words,  # 待提取数据
                  seq_length=args.chn_senti_seq_len,  # 序列长度
                  save_path=os.path.join(args.save_path, 'ChnSentiCorp/senti_test.npy')  # 提取的词向量保存地址
                  )
'''
'''
data_path = './../data/raw data/ChnSentiCorp/train_set.csv'
stopwords_path = './../resources/哈工大停用词表.txt'
train, senti_train = w2v_prep(data_path=data_path, stopwords_path=stopwords_path, lex_basepath=args.lexicon_basepath,
                              lexicon_names=args.lexicon_names, w2v_path=args.word2vec_path, seq_len=108,
                              senti_seq_len=24)

print(train.shape, train)
print(senti_train.shape, senti_train)
'''
# nlpcc 2014 task2 cn
'''
args = utils_parser()

data = pd.read_csv(os.path.join('./../data/raw data/NLPCC 2014 Task2/cn', 'test_set.csv'))  # 数据 ---

extract_embedding(model_path=args.roberta_path,  # 预训练语言模型地址
                  data=data.senti_words,  # 待提取数据 ---
                  seq_length=args.senti_seq_len,  # 序列长度 ---
                  save_path=os.path.join(args.save_path, 'NLPCC 2014 Task2/cn/senti_test.npy')  # 提取的词向量保存地址 ---
                  )
'''
# nlpcc 2014 task2 en
'''
args = utils_parser()
nlpcc_en_args = nlpcc_en_parser()

data = pd.read_csv(os.path.join('./../data/raw data/NLPCC 2014 Task2/en', 'test_set.csv'))  # 数据 --------

extract_embedding(model_path=args.bert_path_en,  # 预训练语言模型地址
                  data=data.review,  # 待提取数据 ---------
                  seq_length=nlpcc_en_args.seq_len,  # 序列长度 用于填充或截取 -----------
                  save_path=os.path.join(nlpcc_en_args.data_path, 'test.npy')  # 提取的词向量保存地址 ------------
                  )
'''

# MR

args = utils_parser()
mr_args = mr_parser()

data = pd.read_csv(os.path.join('./../data/raw data/MR', 'test_set.csv'))  # 数据 --------

extract_embedding(model_path=args.bert_path_en,  # 预训练语言模型地址
                  data=data.review,  # 待提取数据 ---------
                  seq_length=mr_args.seq_len,  # 序列长度 用于填充或截取 -----------
                  save_path=os.path.join(mr_args.data_path, 'test.npy')  # 提取的词向量保存地址 ------------
                  )
