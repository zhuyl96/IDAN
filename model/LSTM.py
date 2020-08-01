# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 20:52 
@Author : zhuyl
@File : IDAN.py
@Desc: 

"""

from keras.layers import Input, Embedding
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from model.calculate_f1_ import Metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from config.config import *
from preprocessing.utils import w2v_prep_, w2v_prep_en_


class LSTMEX:
    def __init__(self, args, hidden_size=512, dropout=0.1, loss='categorical_crossentropy', ):
        """

        :param args:
        :param attention_size:
        :param hidden_size:
        :param n_heads:
        :param dropout:
        :param sentence:
        :param sentiment:
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
        word_dict, embedding_matrix, train, test = w2v_prep_(data_path=args.data_path_w2v,
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
        emb = Embedding(len(word_dict) + 1,
                        self.emb_size, weights=[embedding_matrix],
                        input_length=self.seq_len,
                        trainable=True,
                        name="Sentence_Emb")(input)

        # LSTM
        lstm = LSTM(units=self.hidden_size,
                    return_sequences=False,
                    kernel_regularizer=l2(l=self.l2_reg),
                    bias_regularizer=l2(l=self.l2_reg),
                    name='Sentence_LSTM')(emb)

        dropout = Dropout(rate=self.dropout, name='Dropout')(lstm)

        output = Dense(units=self.categories_num,
                       activation=self.classifier,
                       kernel_regularizer=l2(l=self.l2_reg),
                       bias_regularizer=l2(l=self.l2_reg),
                       name='Softmax')(dropout)  # classifier

        model = Model(inputs=input, outputs=output, name='LSTM')

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
    args = nlpcc_parser()
    for i in range(12):
        print("----------------------------------------------------------------------------")
        print("----------------------------------", i + 1, "---------------------------------------")
        chn = LSTMEX(args=args, )
        chn.model(args)
