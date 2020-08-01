# -*- coding: utf-8 -*-
"""
@Time : 2020/4/12 16:40 
@Author : zhuyl
@File : visualization.py 
@Desc: 

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
from keras_multi_head import MultiHeadAttention
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from keract import get_activations



# set GPU memory
if 'tensorflow' == K.backend():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


def get_activations(model, inputs, print_shape_only=True, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
        print(outputs)
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    print(funcs)
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


# X_test = np.load(file='./reviews_test_300.npy')
# sent_test = np.load(file='./sentiment_test_50.npy')

model = load_model('./model.h5', custom_objects={'MultiHeadAttention': MultiHeadAttention})
print(model.summary)
a=model.predict()

'''
attention_vectors = []

sentence_input = X_test[0]
sentiment_input = sent_test[0]

activations = get_activations(model, [sentence_input, sentiment_input], layer_name='attention_weight_sentence')
attention_vec = np.mean(activations[0], axis=0).squeeze()  # squeeze 表示去除数组中单维度条目（把shape中为1的维度去掉）
print('attention =', attention_vec)
assert (np.sum(attention_vec) - 1.0) < 1e-5
# attention_vectors.append(attention_vec)

# attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
# plot part.

pd.DataFrame(attention_vec, columns=['attention (%)']).plot(kind='bar',
                                                            title='Attention Mechanism as '
                                                                  'a function of input'
                                                                  ' dimensions.')
plt.show()
print(activations[0])
'''
