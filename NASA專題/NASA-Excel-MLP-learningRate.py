#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ ="FU-LONG,SHI"


import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
import myfun
plt.style.use('seaborn-poster')

df = pd.read_excel("NASA(清洗後).xlsx",header=0)
print(df.head(5)) #印出前五筆資料
#   Pandas  x 轉 numpy
x=df.to_numpy()
print(x)

print("====下載資料==============")
col = ['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude']
col_target=['hazardous']


print("====讀取資料==標準化============")

train_x, test_x, train_y, test_y,scaler=myfun.ML_read_dataframe_標準化("NASA(清洗後).xlsx", col, col_target)
print("外型大小",train_x.shape,test_x.shape,train_y.shape,test_y.shape)
print("前面幾筆:",train_x)


dim=5 #x 5個參考訓練欄位
category=2  # y答案2種


train_y2=tf.keras.utils.to_categorical(train_y, num_classes=(category))
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=(category))

print("train_x[:4]",train_x[:4])
print("train_y[:4]",train_y[:4])
print("train_y2[:4]",train_y2[:4])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=100,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

opti1=tf.keras.optimizers.Adadelta(lr=0.0001)     # Adadelta
opti1=tf.keras.optimizers.Adam(lr=0.0001)    # 使用Adam 移動 0.001  #  內定值 learning_rate=0.001,
opti1=tf.keras.optimizers.Nadam(lr=0.0001)
opti1=tf.keras.optimizers.SGD(lr=0.0001)     # 梯度下降

model.compile(optimizer=opti1,
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy','mse','mae'])#+mse
model.fit(train_x, train_y2,
          epochs=200,
          batch_size=100)

#測試
score = model.evaluate(test_x, test_y2, batch_size=128)
print("score:",score)

predict = model.predict(test_x)
print("Ans:",np.argmax(predict,axis=-1))






