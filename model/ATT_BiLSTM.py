# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 20:52 
@Author : zhuyl
@File : IDAN.py
@Desc: 

"""

from keras.layers import Input, Embedding, Bidirectional, dot, concatenate, GlobalAveragePooling1D
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras_multi_head import MultiHeadAttention
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from model.calculate_f1_ import Metrics
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from config.config import *
from preprocessing.utils import w2v_prep_, w2v_prep_en_


class GlobalAttention:
    def __init__(self, args, attention_size):
        """

        :param args:
        :param attention_size:
        """
        self.args = args
        self.attention_size = attention_size
        self.l2_reg = args.l2_reg

    def attention_3d_block(self, hidden_states, sub_name):
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
                                 name='attention_score_vec' + sub_name)(hidden_states)
        # score_first_part dot last_hidden_state => attention_weights

        # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
        h_t = Lambda(lambda x: x[:, -1, :],
                     output_shape=(hidden_size,),
                     name='last_hidden_state' + sub_name)(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score' + sub_name)
        attention_weights = Activation('softmax', name='attention_weight' + sub_name)(score)

        # (batch_size, time_steps，hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector' + sub_name)
        pre_activation = concatenate([context_vector, h_t], name='attention_output' + sub_name)
        attention_vector = Dense(units=self.attention_size,
                                 use_bias=False,
                                 activation='tanh',
                                 kernel_regularizer=l2(l=self.l2_reg),
                                 name='attention_vector' + sub_name)(pre_activation)
        return attention_vector


class ATT_BiLSTM(GlobalAttention):
    def __init__(self, args, attention_size=512, hidden_size=256, dropout=0.1, n_heads=8, sentence='_sentence',
                 loss='categorical_crossentropy', ):
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
        super(ATT_BiLSTM, self).__init__(args, attention_size)

        self.args = args
        self.seq_len = args.seq_len_w2v
        self.attention_size = attention_size

        self.emb_size = args.emb_size_w2v
        self.categories_num = args.categories_num
        self.l2_reg = args.l2_reg
        self.batch_size = args.batch_size
        # self.stopwords_path = args.stopwords_path
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.sentence = sentence
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

        # BiLSTM
        bilstm = Bidirectional(layer=LSTM(units=self.hidden_size,
                                          return_sequences=True,
                                          kernel_regularizer=l2(l=self.l2_reg),
                                          bias_regularizer=l2(l=self.l2_reg),
                                          name='Sentence_BiLSTM'))(emb)

        # sentence global attention
        multi_att = MultiHeadAttention(head_num=self.n_heads,
                                       kernel_regularizer=l2(l=self.l2_reg),
                                       bias_regularizer=l2(l=self.l2_reg),
                                       name='Sentence_MultiAtt')([bilstm, bilstm, bilstm])

        pool = GlobalAveragePooling1D(name="GlobalAveragePooling")(multi_att)

        global_att = super().attention_3d_block(hidden_states=bilstm, sub_name=self.sentence)

        dropout = Dropout(rate=self.dropout, name='Dropout')(pool)

        output = Dense(units=self.categories_num,
                       activation=self.classifier,
                       kernel_regularizer=l2(l=self.l2_reg),
                       bias_regularizer=l2(l=self.l2_reg),
                       name='Softmax')(dropout)  # classifier

        model = Model(inputs=input, outputs=output, name='ATT-BiLSTM')

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        model.summary()

        f1_metrics = Metrics()
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=2)

        model.fit(x=train,
                  y=train_label,
                  shuffle=True,
                  validation_data=(test, test_label),
                  callbacks=[f1_metrics],
                  epochs=10,
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
        chn = ATT_BiLSTM(args=args, )
        chn.model(args)
