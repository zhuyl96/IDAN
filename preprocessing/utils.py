# -*- coding: utf-8 -*-
"""
@Time : 2020/1/1 17:26 
@Author : zhuyl
@File : utils.py
@Desc: 

"""

import numpy as np
import pandas as pd
import jieba
import os
from keras.preprocessing.sequence import pad_sequences
from keras_bert import extract_embeddings
import nltk
import gensim
from collections import defaultdict
import tensorflow as tf


def load_user_lexicon(basepath, lexicon_names):
    """

    :param basepath: 用户词典基础路径
    :param lexicon_names: 词典名字列表
    :return: 三种词典的列表
    """
    lexicons = []
    for lexicon_name in lexicon_names:
        path = os.path.join(basepath, lexicon_name) + '.txt'
        lexicons.append([line.strip() for line in open(path, 'r', encoding='utf-8').readlines()])
        jieba.load_userdict(path)  # 加载为用户词典

    sentiment_words, adverb_words, nagetive_words = [], [], []
    for lexicon in lexicons:
        if 0 <= lexicons.index(lexicon) < 2:
            # 正负情感词
            for word in lexicon:
                sentiment_words.append(word)
        elif lexicons.index(lexicon) == 2:
            # 程度副词
            for word in lexicon:
                adverb_words.append(word)
        else:
            # 否定词
            for word in lexicon:
                nagetive_words.append(word)

    return sentiment_words, adverb_words, nagetive_words


def load_user_lexicon_en(basepath, lexicon_names):
    lexicons = []
    for lexicon_name in lexicon_names:
        path = os.path.join(basepath, lexicon_name) + '.txt'
        lexicons.append([line.strip() for line in open(path, 'r', encoding='utf-8').readlines()])

    sentiment_words, adverb_words = [], []
    for l in lexicons:
        if 0 <= lexicons.index(l) < 2:
            # 正负面情感词
            for word in l:
                sentiment_words.append(word)
        elif lexicons.index(l) == 2:
            # 程度副词
            for word in l:
                adverb_words.append(word)

    sentiment_words = [word.lower() for word in sentiment_words]
    adverb_words = [word.lower() for word in adverb_words]
    return sentiment_words, adverb_words


def extract_senti_words(sentiment_words, adverb_words, negative_words, text):
    """
    :param sentiment_words: 情感词典列表
    :param adverb_words:  程度副词列表
    :param negative_words:  否定词列表
    :param text:  待提取的文本
    :return: 提取之后的文本
    """
    pad = ['pad', 'pad']
    seg_text = list(jieba.cut(text, cut_all=False))
    seg_text.extend(pad)
    text = []

    for index, word in enumerate(seg_text):
        if word in sentiment_words:
            # rule 1
            text.append(word)

        elif word in negative_words:
            # rule 2
            if seg_text[index + 1] in sentiment_words:
                text.append(word)
            elif (seg_text[index + 1] in adverb_words) and (seg_text[index + 2] in sentiment_words):
                text.append(word)

        elif word in adverb_words:
            # rule 3
            if seg_text[index + 1] in sentiment_words:
                text.append(word)
            elif (seg_text[index + 1] in negative_words) and (seg_text[index + 2] in sentiment_words):
                text.append(word)

    text = "".join(text)
    return text


def extract_senti_words_en(sentiment_words, adverb_words, text):
    sent = {}
    for sent_word in sentiment_words:  # 情感词
        index = -1
        index = text.find(sent_word)
        if index != -1:
            sent[sent_word] = index  # 先将所有情感词添加到字典中

    for adv_word in adverb_words:  # 程度副词＋情感词的组合
        index = -1
        index = text.find(adv_word)
        if index != -1:
            seged_text = text[index:]
            for sent_word_ in sentiment_words:  # 若程度副词后跟着一个情感词
                index2 = seged_text.find(sent_word_)
                if index2 == len(adv_word) + 1:
                    sent[adv_word] = index
                    sent[sent_word_] = index2 + index

    sent = sorted(zip(sent.values(), sent.keys()))  # zip() 生成字典 再用sort对字典按value(也就是单词的index进行排序)
    sent = ' '.join(list(dict(sent).values()))
    return sent


def extract_embedding(data, model_path, seq_length, save_path):
    """
    :param data: 待提取的数据
    :param model_path: roberta模型所在目录
    :param seq_length: 序列长度
    :param save_path: 提取到的npy保存的地址
    :param pad_path: 所用的填充向量地址
    :param mode: 模式
    :return:
    """
    reviews = []
    for review in data:
        reviews.append(str(review))

    embeddings = extract_embeddings(model_path, reviews, batch_size=4)
    # 之前使用随机初始化的方法填充是错的 得用全零向量填充
    pad = np.zeros((1, 768))
    # pad = np.load(file=pad_path)
    for i, array in enumerate(embeddings):
        if array.shape[0] > seq_length:
            # 序列长度大于最大长度的只要前max个嵌入向量
            embeddings[i] = array[:seq_length, ]
        else:
            # 不足的进行填充
            pads = np.tile(pad, seq_length - array.shape[0])
            pads = pads.reshape((seq_length - array.shape[0], 768))
            embeddings[i] = np.row_stack((array, pads))
    embeddings = np.asarray(embeddings)
    np.save(file=save_path, arr=embeddings)


def w2v_prep(data_path, lex_basepath, lexicon_names, w2v_path, seq_len, senti_seq_len):
    """

    :param data_path:
    :param stopwords_path:
    :param lex_basepath:
    :param lexicon_names:
    :param w2v_path:
    :param seq_len:
    :param senti_seq_len:
    :return:
    """
    train_data = pd.read_csv(os.path.join(data_path, 'train_set.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_set.csv'))

    # stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]

    train_data['review_w2v'] = train_data['review'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])
    test_data['review_w2v'] = test_data['review'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])

    lexicons = []
    for lexicon_name in lexicon_names:
        path = os.path.join(lex_basepath, lexicon_name) + '.txt'
        lexicons.append([line.strip() for line in open(path, 'r', encoding='utf-8').readlines()])
        jieba.load_userdict(path)  # 加载为用户词典
    train_data['senti_w2v'] = train_data['senti_words'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])
    test_data['senti_w2v'] = test_data['senti_words'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])

    # 建立数据集的词频词典 和 词表
    attributes = ['review_w2v', 'senti_w2v']
    word_dict = {}
    vocab = defaultdict(float)
    word_dict.update({'PAD': 0, 'UNK': 1})  # [PAD] [UNK]
    n = 2
    for attribute in attributes:
        for text in train_data[attribute]:
            for i in text:
                if i not in word_dict.keys():
                    word_dict[i] = n
                    n += 1
                if i in vocab:
                    vocab[i] += 1
                else:
                    vocab[i] = 1
        for text in test_data[attribute]:
            for i in text:
                if i not in word_dict.keys():
                    word_dict[i] = n
                    n += 1
                if i in vocab:
                    vocab[i] += 1
                else:
                    vocab[i] = 1

    w2v_model = {}
    f = open(w2v_path, 'rb')
    for line in f:
        values = line.split()
        word = values[0].decode()
        coefs = np.asarray(values[1:], dtype='float32')
        w2v_model[word] = coefs
    f.close()

    # 词表中所有词的嵌入矩阵
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_dict) + 1, 300))
    embedding_matrix[0] = np.zeros(shape=(1, 300))  # [PAD]=embedding_matrix[0]  [UNK]=embedding_matrix[1]
    for word, i in word_dict.items():
        if word in w2v_model.keys():
            embedding_vector = w2v_model.get(word)
            embedding_matrix[i] = embedding_vector

    # train_data 单词转换成序列
    temp = []
    for attribute in attributes:
        for text in train_data[attribute]:
            indexs = []
            for word in text:
                if not (vocab[word] < 2 and word not in w2v_model.keys()):  # 要在word2vec中的词(UNK需词频大于2)
                    index = word_dict.get(word)
                    indexs.append(index)
            temp.append(indexs)
    temp = np.asarray(temp)
    # 前一半是句子 后一半是情感信息
    train = pad_sequences(temp[:int(len(temp) / 2)], maxlen=seq_len, padding='post')  # PAD
    senti_train = pad_sequences(temp[int(len(temp) / 2):], maxlen=senti_seq_len, padding='post')

    # test_data
    temp = []
    for attribute in attributes:
        for text in test_data[attribute]:
            indexs = []
            for word in text:
                if not (vocab[word] < 2 and word not in w2v_model.keys()):
                    # 要在word2vec中的词(UNK需词频大于2)
                    index = word_dict.get(word)
                    indexs.append(index)
            temp.append(indexs)
    temp = np.asarray(temp)
    test = pad_sequences(temp[:int(len(temp) / 2)], maxlen=seq_len, padding='post')
    senti_test = pad_sequences(temp[int(len(temp) / 2):], maxlen=senti_seq_len, padding='post')

    return word_dict, embedding_matrix, train, senti_train, test, senti_test


# 有情感倾向信息的baseline使用
def w2v_prep_en(data_path, w2v_path, seq_len, senti_seq_len):
    train_data = pd.read_csv(os.path.join(data_path, 'train_set.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_set.csv'))
    w2v_path = os.path.join(w2v_path, 'GoogleNews-vectors-negative300.bin')

    # 建立数据集的词频词典
    vocab = defaultdict(float)
    for i in train_data['review']:
        for j in i.split():
            if j in vocab:
                vocab[j] += 1
            else:
                vocab[j] = 1
    for i in test_data['review']:
        for j in i.split():
            if j in vocab:
                vocab[j] += 1
            else:
                vocab[j] = 1

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    train_data['review_w2v'] = train_data['review'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])
    train_data['senti_w2v'] = train_data['senti_words'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])
    test_data['review_w2v'] = test_data['review'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])
    test_data['senti_w2v'] = test_data['senti_words'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])

    # 建立word_dict
    attributes = ['review_w2v', 'senti_w2v']
    word_dict = {}
    word_dict.update({'PAD': 0, 'UNK': 1})  # [PAD] [UNK] 没有使用
    n = 2
    for attribute in attributes:
        for text in train_data[attribute]:
            for i in text:
                if i not in word_dict.keys():
                    word_dict[i] = n
                    n += 1
        for text in test_data[attribute]:
            for i in text:
                if i not in word_dict.keys():
                    word_dict[i] = n
                    n += 1

    # 词表中所有词的嵌入矩阵
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_dict) + 1, 300))
    embedding_matrix[0] = np.zeros(shape=(1, 300))  # [PAD]=embedding_matrix[0]  [UNK]=embedding_matrix[1]
    for word, i in word_dict.items():
        if word in w2v_model.vocab:
            embedding_vector = w2v_model[word]
            embedding_matrix[i] = embedding_vector

    # train_data 单词转换成序列
    temp = []
    for attribute in attributes:
        for text in train_data[attribute]:
            indexs = []
            for word in text:
                if not (vocab[word] < 2 and word not in w2v_model.vocab):  # 要在word2vec中的词(UNK需词频大于2)
                    index = word_dict.get(word)
                    indexs.append(index)
            temp.append(indexs)
    temp = np.asarray(temp)
    # 前一半是句子 后一半是情感信息
    train = pad_sequences(temp[:int(len(temp) / 2)], maxlen=seq_len, padding='post')  # PAD
    senti_train = pad_sequences(temp[int(len(temp) / 2):], maxlen=senti_seq_len, padding='post')

    # test_data
    temp = []
    for attribute in attributes:
        for text in test_data[attribute]:
            indexs = []
            for word in text:
                if not (vocab[word] < 2 and word not in w2v_model.vocab):
                    # 要在word2vec中的词(UNK需词频大于2)
                    index = word_dict.get(word)
                    indexs.append(index)
            temp.append(indexs)
    temp = np.asarray(temp)
    test = pad_sequences(temp[:int(len(temp) / 2)], maxlen=seq_len, padding='post')
    senti_test = pad_sequences(temp[int(len(temp) / 2):], maxlen=senti_seq_len, padding='post')

    return word_dict, embedding_matrix, train, senti_train, test, senti_test


def w2v_prep_(data_path, w2v_path, seq_len):
    """

    :param data_path:
    :param stopwords_path:
    :param w2v_path:
    :param seq_len:
    :return:
    """
    train_data = pd.read_csv(os.path.join(data_path, 'train_set.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_set.csv'))
    # 停用词
    # stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]

    train_data['review_w2v'] = train_data['review'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])
    test_data['review_w2v'] = test_data['review'].map(
        lambda x: [word for word in jieba.cut(str(x), cut_all=False)])

    # 建立数据集的词频词典 和 词表
    attribute = 'review_w2v'
    word_dict = {}
    vocab = defaultdict(float)
    word_dict.update({'PAD': 0, 'UNK': 1})  # [PAD] [UNK]
    n = 2
    for text in train_data[attribute]:
        for i in text:
            if i not in word_dict.keys():
                word_dict[i] = n
                n += 1
            if i in vocab:
                vocab[i] += 1
            else:
                vocab[i] = 1
    for text in test_data[attribute]:
        for i in text:
            if i not in word_dict.keys():
                word_dict[i] = n
                n += 1
            if i in vocab:
                vocab[i] += 1
            else:
                vocab[i] = 1

    # 从文件读取词嵌入
    w2v_model = {}
    f = open(w2v_path, 'rb')
    for line in f:
        values = line.split()
        word = values[0].decode()
        coefs = np.asarray(values[1:], dtype='float32')
        w2v_model[word] = coefs
    f.close()

    # 词表中所有词的嵌入矩阵
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_dict) + 1, 300))
    embedding_matrix[0] = np.zeros(shape=(1, 300))  # [PAD]=embedding_matrix[0]  [UNK]=embedding_matrix[i]
    for word, i in word_dict.items():
        if word in w2v_model.keys():
            embedding_vector = w2v_model.get(word)
            embedding_matrix[i] = embedding_vector

    # train_data 单词转换成序列
    temp = []
    for text in train_data[attribute]:
        indexs = []
        for word in text:
            if not (vocab[word] < 2 and word not in w2v_model.keys()):
                # 要在word2vec中的词(UNK需词频大于2)
                index = word_dict.get(word)
                indexs.append(index)
        temp.append(indexs)
    temp = np.asarray(temp)
    train = pad_sequences(temp, maxlen=seq_len, padding='post')

    # test data
    temp = []
    for text in test_data[attribute]:
        indexs = []
        for word in text:
            if not (vocab[word] < 2 and word not in w2v_model.keys()):
                # 要在word2vec中的词(UNK需词频大于2)
                index = word_dict.get(word)
                indexs.append(index)
        temp.append(indexs)
    temp = np.asarray(temp)
    test = pad_sequences(temp, maxlen=seq_len, padding='post')

    return word_dict, embedding_matrix, train, test


# 没有情感倾向信息的baseline使用
def w2v_prep_en_(data_path, w2v_path, seq_len):
    train_data = pd.read_csv(os.path.join(data_path, 'train_set.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_set.csv'))
    w2v_path = os.path.join(w2v_path, 'GoogleNews-vectors-negative300.bin')

    # 建立数据集的词频词典
    vocab = defaultdict(float)
    for i in train_data['review']:
        for j in i.split():
            if j in vocab:
                vocab[j] += 1
            else:
                vocab[j] = 1
    for i in test_data['review']:
        for j in i.split():
            if j in vocab:
                vocab[j] += 1
            else:
                vocab[j] = 1

    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)  # 加载Word2Vec

    train_data['review_w2v'] = train_data['review'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])
    test_data['review_w2v'] = test_data['review'].map(lambda x: [word for word in nltk.word_tokenize(str(x))])

    # 建立word_dict
    attribute = 'review_w2v'
    word_dict = {}
    word_dict.update({'PAD': 0, 'UNK': 1})  # [PAD] [UNK]
    n = 2
    for text in train_data[attribute]:
        for i in text:
            if i not in word_dict.keys():
                word_dict[i] = n
                n += 1
    for text in test_data[attribute]:
        for i in text:
            if i not in word_dict.keys():
                word_dict[i] = n
                n += 1

    # 词表中所有词的嵌入矩阵
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(word_dict) + 1, 300))
    embedding_matrix[0] = np.zeros(shape=(1, 300))  # [PAD]=embedding_matrix[0]  [UNK]=embedding_matrix[1]
    for word, i in word_dict.items():
        if word in w2v_model.vocab:
            embedding_vector = w2v_model[word]
            embedding_matrix[i] = embedding_vector
        # 否则 原来embeddingmatrix[i]就相当于对该单词的初始化 后面再处理UNK

    # train_data 单词转换成序列
    temp = []
    for text in train_data[attribute]:
        indexs = []
        for word in text:
            if not (vocab[word] < 2 and word not in w2v_model.vocab):
                # 要在word2vec中的词(UNK需词频大于2)
                index = word_dict.get(word)
                indexs.append(index)
        temp.append(indexs)
    temp = np.asarray(temp)
    train = pad_sequences(temp, maxlen=seq_len, padding='post')

    # test data
    temp = []
    for text in test_data[attribute]:
        indexs = []
        for word in text:
            if not (vocab[word] < 2 and word not in w2v_model.vocab):
                # 要在word2vec中的词(UNK需词频大于2)
                index = word_dict.get(word)
                indexs.append(index)
        temp.append(indexs)
    temp = np.asarray(temp)
    test = pad_sequences(temp, maxlen=seq_len, padding='post')

    return word_dict, embedding_matrix, train, test


# SVM使用
def ind2vec(embedding_matrix, train, test, ):
    reviews = []
    for i in train:
        review = []
        m = 0
        for j in i:
            if m == 0 or j != 0:
                review.append(embedding_matrix[j])
            m += 1
            # 按列求平均 将每条评论转换为[300,1]的大小
        review = np.array(review).mean(axis=0)
        reviews.append(review)
    train = np.array(reviews)

    reviews = []
    for i in test:
        review = []
        m = 0
        for j in i:
            if m == 0 or j != 0:
                review.append(embedding_matrix[j])
            m += 1
            # 按列求平均 将每条评论转换为[300,1]的大小
        review = np.array(review).mean(axis=0)
        reviews.append(review)
    test = np.array(reviews)
    return train, test


def normalizing(x, axis):
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
    normalized = x / (norm)
    return normalized
