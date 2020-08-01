# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 20:52 
@Author : zhuyl
@File : CRNN.py
@Desc: 

"""

from keras.layers import Input, Embedding, Bidirectional, Conv1D, MaxPooling1D, Concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.models import Model
from model.calculate_f1_ import Metrics
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from config.config import *
from preprocessing.utils import w2v_prep_, w2v_prep_en_
from keras.constraints import maxnorm


class CRNN:
    def __init__(self, args, hidden_size=128, dropout=0.1, loss='categorical_crossentropy', ):
        """

        :param args:
        :param hidden_size:
        :param dropout:
        :param loss:
        """

        self.args = args
        self.seq_len = args.seq_len_w2v

        self.emb_size = args.emb_size_w2v
        self.categories_num = args.categories_num
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size

        # self.stopwords_path = args.stopwords_path
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.loss = loss
        self.classifier = 'softmax'
        self.optimizer = Adam(lr=args.learning_rate)

    def load_data(self, args):
        word_dict, embedding_matrix, train, test = w2v_prep_en_(data_path=args.data_path_w2v,

                                                                w2v_path=args.word2vec_path,
                                                                seq_len=self.seq_len, )
        print(train.shape, test.shape, )
        train_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'train_set.csv'))
        train_label = to_categorical(np.asarray(list(train_label.label)))

        test_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'test_set.csv'))
        test_label = to_categorical(np.asarray(list(test_label.label)))

        return word_dict, embedding_matrix, train, train_label, test, test_label

    def model(self, args):
        """

        :param args:
        :return:
        """
        word_dict, embedding_matrix, train, train_label, test, test_label = self.load_data(args)

        # Input
        input = Input(shape=(self.seq_len,), name="Input")
        emb = Embedding(len(word_dict) + 1, self.emb_size, weights=[embedding_matrix], input_length=self.seq_len,
                        trainable=True, name="Sentence_Emb")(input)
        emb = Dropout(0.50)(emb)

        conv4 = Conv1D(filters=200, kernel_size=4, padding='valid', activation='relu', strides=1, name='Conv4')(emb)
        maxConv4 = MaxPooling1D(pool_size=2, name='maxConv4')(conv4)
        conv5 = Conv1D(filters=200, kernel_size=5, padding='valid', activation='relu', strides=1, name='Conv5')(emb)
        maxConv5 = MaxPooling1D(pool_size=2, name='maxConv5')(conv5)

        merge = Concatenate(name='Concatenate')([maxConv4, maxConv5])

        x = Dropout(0.15)(merge)
        x = GRU(units=100)(x)
        x = Dense(400, activation='relu', kernel_initializer='he_normal',
                  kernel_constraint=maxnorm(3), bias_constraint=maxnorm(3),
                  name='mlp')(x)
        x = Dropout(rate=self.dropout, name='drop')(x)

        output = Dense(units=self.categories_num, kernel_initializer='he_normal',
                       activation=self.classifier, name='output')(x)

        model = Model(inputs=input, outputs=output, name='CRNN')

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        model.summary()

        f1_metrics = Metrics()

        model.fit(x=train,
                  y=train_label,
                  shuffle=True,
                  validation_data=(test, test_label),
                  callbacks=[f1_metrics],
                  epochs=30,
                  batch_size=self.batch_size,
                  verbose=2)

        del model


# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

if __name__ == '__main__':
    args = nlpcc_en_parser()
    for i in range(12):
        print("----------------------------------------------------------------------------")
        print("----------------------------------", i + 1, "---------------------------------------")
        chn = CRNN(args=args, )
        chn.model(args)
