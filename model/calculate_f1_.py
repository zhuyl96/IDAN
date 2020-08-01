# -*- coding: utf-8 -*-
"""
@Time : 2020/1/2 22:04 
@Author : zhuyl
@Email : ylzhu1996@qq.com
@File : calculate_f1.py 
@Software: PyCharm
@Desc: 

"""

from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        """

        :param logs:
        :return:
        """
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        """
        :param epoch:
        :param logs:
        :return:
        """
        y_pred = (self.model.predict(x=self.validation_data[0], batch_size=16, verbose=2))  # 这里只有一个输入
        y_pred = [list(i).index(max(i)) for i in y_pred]
        y_true = [i[1] for i in self.validation_data[1]]


        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        # f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        print('f1_macro: ', np.around(f1_macro, 4))
        return
