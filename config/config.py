# -*- coding: utf-8 -*-
"""
@Time : 2019/12/30 22:27 
@Author : zhuyl
@File : config.py
@Desc: 

"""

import os
import argparse


def chn_parser(emb_size=768, batch_size=64, epochs=10, learning_rate=0.0001, seq_len=300, senti_seq_len=50,
               categories_num=2, l2_reg=0.00001, seq_len_w2v=186, senti_seq_len_w2v=28, emb_size_w2v=300, ):
    parser = argparse.ArgumentParser(description='preprocessing for ChnSentiCorp')
    parser.add_argument('--data_basepath', type=str, default='./../data/raw data/ChnSentiCorp',
                        help="chnsenticorp path, ./../ 表示当前目录的上级目录")
    parser.add_argument('--emb_size', type=int, default=emb_size, help="word embedding dimension")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="batch size for training")
    parser.add_argument('--epochs', type=int, default=epochs, help="train epochs")
    parser.add_argument('--learning_rate', type=int, default=learning_rate, help="learning rate")
    parser.add_argument('--seq_len', type=int, default=seq_len, help="max sequence length")
    parser.add_argument('--senti_seq_len', type=int, default=senti_seq_len, help="max sentiment sequence length")
    parser.add_argument('--categories_num', type=int, default=categories_num, help="categories numbers")
    parser.add_argument('--l2_reg', type=int, default=l2_reg, help="l2 regularization")
    parser.add_argument('--logging_path', type=str, default='./../logs/ChnSentiCorp', help="warmmup steps")
    parser.add_argument('--data_path', type=str, default='./../data/npy data/ChnSentiCorp',
                        help="train and test sets path")
    parser.add_argument('--label_path', type=str, default='./../data/raw data/ChnSentiCorp', help="label path")

    parser.add_argument('--seq_len_w2v', type=int, default=seq_len_w2v, help="max sequence length word2vec")
    parser.add_argument('--emb_size_w2v', type=int, default=emb_size_w2v, help="word embedding dimension word2vec")
    parser.add_argument('--senti_seq_len_w2v', type=int, default=senti_seq_len_w2v,
                        help="max sentiment sequence length word2vec")
    parser.add_argument('--data_path_w2v', type=str, default='./../data/raw data/ChnSentiCorp',
                        help="train and test sets path word2vec")
    parser.add_argument('--word2vec_path', type=str,
                        default='./../resources/pre-trained model/word2vec/word2vec-300d.txt',
                        help="word2vec pre-trained model path")
    parser.add_argument('--stopwords_path', type=str, default='./../resources/StopWords.txt', help="stopwords path")
    parser.add_argument('--lexicon_basepath', type=str, default='./../resources/lexicon/cn', help="lexicon base path")
    parser.add_argument('--lexicon_names', type=list,
                        default=['Sentiment(pos)', 'Sentiment(neg)', 'Intensity', 'Negative'],
                        help="the name list of lexicon")

    args = parser.parse_args()
    return args


def utils_parser(seq_len=128, senti_seq_len=24):
    parser = argparse.ArgumentParser(description='utils')
    parser.add_argument('--lexicon_basepath', type=str, default='./../resources/lexicon/cn', help="lexicon base path")
    parser.add_argument('--lexicon_names', type=list,
                        default=['Sentiment(pos)', 'Sentiment(neg)', 'Intensity', 'Negative'],
                        help="the name list of lexicon")
    parser.add_argument('--bert_path_cn', type=str, default='./../resources/pre-trained model/cn bert',
                        help="cn bert pre-trained model path")
    parser.add_argument('--bert_path_en', type=str, default='./../resources/pre-trained model/en bert',
                        help="en bert pre-trained model path")
    parser.add_argument('--word2vec_path', type=str,
                        default='./../resources/pre-trained model/word2vec/word2vec-300d.txt',
                        help="word2vec pre-trained model path")
    parser.add_argument('--save_path', type=str, default='./../data/npy data', help="npy save path")
    parser.add_argument('--seq_len', type=int, default=seq_len, help="max review length")
    parser.add_argument('--senti_seq_len', type=int, default=senti_seq_len, help="max sentiment length")
    parser.add_argument('--pad_npy_path', type=str, default='./../data/npy data/pad.npy', help="pad npy path")
    parser.add_argument('--stopwords_path', type=str, default='./../resources/StopWords.txt', help="stopwords path")
    args = parser.parse_args()
    return args


def nlpcc_parser(seq_len=128, senti_seq_len=24, seq_len_w2v=82, senti_seq_len_w2v=14, emb_size=768, categories_num=2,
                 l2_reg=0.00001, epochs=20, batch_size=64, learning_rate=0.0001, emb_size_w2v=300):
    parser = argparse.ArgumentParser(description='preprocessing for NLPCC 2014')
    parser.add_argument('--data_path', type=str, default='./../data/npy data/NLPCC 2014 Task2/cn',
                        help="nlpcc 2014 data path")
    parser.add_argument('--seq_len', type=int, default=seq_len, help="max sequence length")
    parser.add_argument('--senti_seq_len', type=int, default=senti_seq_len, help="max sentiment sequence length")
    parser.add_argument('--seq_len_w2v', type=int, default=seq_len_w2v, help="max sequence length")
    parser.add_argument('--senti_seq_len_w2v', type=int, default=senti_seq_len_w2v,
                        help="max sentiment sequence length")
    parser.add_argument('--stopwords_path', type=str, default='./../resources/StopWords.txt', help="stopwords path")
    parser.add_argument('--word2vec_path', type=str,
                        default='./../resources/pre-trained model/word2vec/word2vec-300d.txt',
                        help="word2vec pre-trained model path")
    parser.add_argument('--label_path', type=str, default='./../data/raw data/NLPCC 2014 Task2/cn', help="label path")
    parser.add_argument('--emb_size', type=int, default=emb_size, help="word embedding dimension")
    parser.add_argument('--categories_num', type=int, default=categories_num, help="categories numbers")
    parser.add_argument('--l2_reg', type=int, default=l2_reg, help="l2 regularization")
    parser.add_argument('--epochs', type=int, default=epochs, help="train epochs")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="batch size for training")
    parser.add_argument('--learning_rate', type=int, default=learning_rate, help="learning rate")
    parser.add_argument('--lexicon_basepath', type=str, default='./../resources/lexicon/cn', help="lexicon base path")
    parser.add_argument('--lexicon_names', type=list,
                        default=['Sentiment(pos)', 'Sentiment(neg)', 'Intensity', 'Negative'],
                        help="the name list of lexicon")
    parser.add_argument('--data_path_w2v', type=str, default='./../data/raw data/NLPCC 2014 Task2/cn',
                        help="nlpcc 2014 data path word2vec")
    parser.add_argument('--emb_size_w2v', type=int, default=emb_size_w2v, help="word embedding dimension word2vec")
    args = parser.parse_args()
    return args


def nlpcc_en_parser(seq_len=300, senti_seq_len=34, seq_len_w2v=300, senti_seq_len_w2v=32, emb_size=768,
                    categories_num=2, l2_reg=0.00001, batch_size=64, learning_rate=0.0001, emb_size_w2v=300):
    parser = argparse.ArgumentParser(description='preprocessing for NLPCC 2014 en')
    parser.add_argument('--data_path', type=str, default='./../data/npy data/NLPCC 2014 Task2/en',
                        help="nlpcc 2014 en npy data path")
    parser.add_argument('--seq_len', type=int, default=seq_len, help="max sequence length")
    parser.add_argument('--senti_seq_len', type=int, default=senti_seq_len, help="max sentiment sequence length")
    parser.add_argument('--label_path', type=str, default='./../data/raw data/NLPCC 2014 Task2/en', help="label path")
    parser.add_argument('--emb_size', type=int, default=emb_size, help="word embedding dimension")
    parser.add_argument('--categories_num', type=int, default=categories_num, help="categories numbers")
    parser.add_argument('--l2_reg', type=int, default=l2_reg, help="l2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="batch size for training")
    parser.add_argument('--learning_rate', type=int, default=learning_rate, help="learning rate")
    parser.add_argument('--lexicon_basepath', type=str, default='./../resources/lexicon/en', help="lexicon base path")
    parser.add_argument('--lexicon_names', type=list, default=['Sentiment(pos)', 'Sentiment(neg)', 'Intensity'],
                        help="the name list of lexicon")

    parser.add_argument('--data_path_w2v', type=str, default='./../data/raw data/NLPCC 2014 Task2/en',
                        help="nlpcc 2014 data path word2vec")
    parser.add_argument('--seq_len_w2v', type=int, default=seq_len_w2v, help="max sequence length")
    parser.add_argument('--senti_seq_len_w2v', type=int, default=senti_seq_len_w2v,
                        help="max sentiment sequence length")
    parser.add_argument('--emb_size_w2v', type=int, default=emb_size_w2v, help="word embedding dimension word2vec")
    parser.add_argument('--word2vec_path', type=str, default='./../resources/pre-trained model/word2vec',
                        help="word2vec pre-trained model path")

    args = parser.parse_args()
    return args


def mr_parser(seq_len=34, senti_seq_len=7, seq_len_w2v=32, senti_seq_len_w2v=6, emb_size=768, categories_num=2,
              l2_reg=0.00001, batch_size=64, learning_rate=0.0001, emb_size_w2v=300):
    parser = argparse.ArgumentParser(description='preprocessing for MR')
    parser.add_argument('--data_path', type=str, default='./../data/npy data/MR', help="nlpcc 2014 en npy data path")
    parser.add_argument('--seq_len', type=int, default=seq_len, help="max sequence length")
    parser.add_argument('--senti_seq_len', type=int, default=senti_seq_len, help="max sentiment sequence length")
    parser.add_argument('--label_path', type=str, default='./../data/raw data/MR', help="label path")
    parser.add_argument('--emb_size', type=int, default=emb_size, help="word embedding dimension")
    parser.add_argument('--categories_num', type=int, default=categories_num, help="categories numbers")
    parser.add_argument('--l2_reg', type=int, default=l2_reg, help="l2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="batch size for training")
    parser.add_argument('--learning_rate', type=int, default=learning_rate, help="learning rate")
    parser.add_argument('--lexicon_basepath', type=str, default='./../resources/lexicon/en', help="lexicon base path")
    parser.add_argument('--lexicon_names', type=list, default=['Sentiment(pos)', 'Sentiment(neg)', 'Intensity'],
                        help="the name list of lexicon")

    parser.add_argument('--data_path_w2v', type=str, default='./../data/raw data/MR',
                        help="nlpcc 2014 data path word2vec")
    parser.add_argument('--seq_len_w2v', type=int, default=seq_len_w2v, help="max sequence length")
    parser.add_argument('--senti_seq_len_w2v', type=int, default=senti_seq_len_w2v,
                        help="max sentiment sequence length")
    parser.add_argument('--emb_size_w2v', type=int, default=emb_size_w2v, help="word embedding dimension word2vec")
    parser.add_argument('--word2vec_path', type=str, default='./../resources/pre-trained model/word2vec',
                        help="word2vec pre-trained model path")

    args = parser.parse_args()
    return args
