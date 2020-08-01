# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 20:52 
@Author : zhuyl
@File : IDAN.py
@Desc: 

"""

from keras.layers import Input, Embedding, Bidirectional, Dropout, Concatenate, dot, concatenate
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras_multi_head import MultiHeadAttention
from model.calculate_f1 import Metrics
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from keras import backend as K
from config.config import *


class GlobalAttention:
    def __init__(self, args, attention_size):
        """
        :param args:
        :param attention_size:
        """
        self.args = args
        self.attention_size = attention_size
        self.l2_reg = args.l2_reg

    def attention_3d_block(self, hidden_states, lstm_last_state, sub_name):
        """
        :param hidden_states:
        :param sub_name:
        :return:
        """
        # hidden_states.shape = (batch_size, time_steps, hidden_size)
        hidden_size = int(hidden_states.shape[2])

        # Inside dense layer
        # hidden_states dot W => score_first_part
        # (batch_size, time_steps, hidden_size) dot (time_steps, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention
        # Luong's multiplicative style score
        score_first_part = Dense(units=hidden_size,
                                 use_bias=False,
                                 kernel_regularizer=l2(l=self.l2_reg),
                                 name='attention_score_vec' + sub_name)(
            hidden_states)  # (batch size, seq len, hidden size)
        # score_first_part dot last_hidden_state => attention_weights

        # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)

        h_t = Lambda(lambda x: x[:, -1, :],
                     output_shape=(hidden_size,),
                     name='last_hidden_state' + sub_name)(lstm_last_state)  # (batch size, 1, hidden size)

        score = dot([score_first_part, h_t], [2, 1], name='attention_score' + sub_name)  # (batch size, seq len)
        attention_weights = Activation('softmax', name='attention_weight' + sub_name)(score)  # (batch size, seq len)
        # (batch_size, time_steps，hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1],
                             name='context_vector' + sub_name)  # (batch size, hidden size)
        pre_activation = concatenate([context_vector, h_t],
                                     name='attention_output' + sub_name)  # (batch size, hidden size x 2)
        attention_vector = Dense(units=self.attention_size,
                                 use_bias=False,
                                 activation='tanh',
                                 kernel_regularizer=l2(l=self.l2_reg),
                                 name='attention_vector' + sub_name)(pre_activation)  # (batch size, attention_size)
        return attention_vector


class IDANModel(GlobalAttention):
    def __init__(self, args, attention_size=512, hidden_size=256, n_heads=8, dropout=0.1, sentence='_sentence',
                 sentiment='_sentiment', loss='categorical_crossentropy', ):
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
        super(IDANModel, self).__init__(args, attention_size)

        self.args = args
        self.seq_len = args.seq_len
        self.senti_seq_len = args.senti_seq_len
        self.emb_size = args.emb_size
        self.categories_num = args.categories_num
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size

        self.attention_size = attention_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.sentence = sentence
        self.sentiment = sentiment
        self.loss = loss
        self.classifier = 'softmax'
        self.optimizer = Adam(lr=args.learning_rate, )
        # self.logging = TensorBoard(log_dir=args.logging_path,
        #                            update_freq='batch',
        #                            batch_size=self.batch_size,
        #                            write_images=True)

    def load_data(self, args):
        print("Load data...")
        train = np.load(file=os.path.join(args.data_path, 'train.npy'))
        senti_train = np.load(file=os.path.join(args.data_path, 'senti_train.npy'))
        test = np.load(file=os.path.join(args.data_path, 'test.npy'))
        senti_test = np.load(file=os.path.join(args.data_path, 'senti_test.npy'))

        print(train.shape, senti_train.shape, test.shape, senti_test.shape)

        train_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'train_set.csv'))
        train_label = to_categorical(np.asarray(list(train_label.label)))
        test_label = pd.read_csv(filepath_or_buffer=os.path.join(args.label_path, 'test_set.csv'))
        test_label = to_categorical(np.asarray(list(test_label.label)))

        return train, senti_train, test, senti_test, train_label, test_label

    def model(self, args):
        """
        :param args:
        :return:
        """
        train, senti_train, test, senti_test, train_label, test_label = self.load_data(args)

        # sentence input
        sentence_input = Input(shape=(self.seq_len, self.emb_size,), name='Sentence_Input')
        # sentiment input
        sentiment_input = Input(shape=(self.senti_seq_len, self.emb_size,), name='Sentiment_Input')

        # sentence BiLSTM
        sentence_lstm = Bidirectional(layer=LSTM(units=self.hidden_size,
                                                 return_sequences=True,
                                                 kernel_regularizer=l2(l=self.l2_reg),
                                                 bias_regularizer=l2(l=self.l2_reg),
                                                 name='Sentence_BiLSTM'))(sentence_input)
        # sentiment BiLSTM
        sentiment_lstm = Bidirectional(layer=LSTM(units=self.hidden_size,
                                                  return_sequences=True,
                                                  kernel_regularizer=l2(l=self.l2_reg),
                                                  bias_regularizer=l2(l=self.l2_reg),
                                                  name='Sentiment_BiLSTM'))(sentiment_input)

        # sentence multi-head attention
        sentence_att = MultiHeadAttention(head_num=self.n_heads,
                                          kernel_regularizer=l2(l=self.l2_reg),
                                          bias_regularizer=l2(l=self.l2_reg),
                                          name='Sentence_MultiAtt')([sentence_lstm, sentiment_lstm, sentiment_lstm])

        # sentiment multi-head attention
        sentiment_att = MultiHeadAttention(head_num=self.n_heads,
                                           kernel_regularizer=l2(l=self.l2_reg),
                                           bias_regularizer=l2(l=self.l2_reg),
                                           name='Sentiment_MultiAtt')([sentiment_lstm, sentence_lstm, sentence_lstm])
        print(sentence_lstm.shape)
        # sentence global attention
        sentence_global_att = super().attention_3d_block(hidden_states=sentence_att, lstm_last_state=sentence_lstm,
                                                         sub_name=self.sentence)
        # sentence global attention
        sentiment_global_att = super().attention_3d_block(hidden_states=sentiment_att, lstm_last_state=sentiment_lstm,
                                                          sub_name=self.sentiment)

        # concatenate sentence and sentiment representation
        merge = Concatenate(name='Concatenate')([sentence_global_att, sentiment_global_att])

        dropout = Dropout(rate=self.dropout, name='Dropout')(merge)

        output = Dense(units=self.categories_num,
                       activation=self.classifier,
                       kernel_regularizer=l2(l=self.l2_reg),
                       bias_regularizer=l2(l=self.l2_reg),
                       name='Softmax')(dropout)  # classifier

        model = Model(inputs=[sentence_input, sentiment_input], outputs=output, name='IDAN')

        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        model.summary()

        f1_metrics = Metrics()

        model.fit(x=[train, senti_train],
                  y=train_label,
                  shuffle=True,
                  validation_data=([test, senti_test], test_label),
                  callbacks=[f1_metrics],
                  epochs=5,
                  batch_size=self.batch_size,
                  verbose=2)

        del model


# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

if __name__ == '__main__':
    args = chn_parser()
    for i in range(12):
        print("----------------------------------------------------------------------------")
        print("----------------------------------", i + 1, "---------------------------------------")
        model = IDANModel(args=args, )
        model.model(args)
