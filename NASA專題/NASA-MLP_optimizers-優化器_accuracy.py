#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "FULONG-SHI"

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import 標準化
import matplotlib.pyplot as plt

import matplotlib
#ravel 轉換為度 一維
plt.style.use('seaborn-poster')
df = pd.read_excel("NASA(清洗後).xlsx",header=0)
print(df.head(5)) #印出前五筆資料
#   Pandas  x 轉 numpy
x=df.to_numpy()
print(x)

print("====下載資料==============")
colX = ['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude']
col_target=['hazardous']

print("====讀取資料==標準化============")

train_x, test_x, train_y, test_y,scaler=標準化.ML_read_dataframe_標準化("NASA(清洗刪減).xlsx", colX, col_target)
print("外型大小",train_x.shape,test_x.shape,train_y.shape,test_y.shape)
print("前面幾筆:",train_x)






dim=5 #x 5個參考訓練欄位
category=2  # y答案2種


train_y2=tf.keras.utils.to_categorical(train_y, num_classes=(category)) #熱編碼 y的標準化
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=(category)) #熱編碼 y的標準化

print("train_x[:4]",train_x[:4])
print("train_y[:4]",train_y[:4])
print("train_y2[:4]",train_y2[:4])
print("test_y[:4]",test_y[:4])
print("test_y2[:4]",test_y2[:4])



def AI_MLP(opti1,train_x , test_x , train_y , test_y,train_y2 , test_y2):
    # 建立模型
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu,
                                    input_dim=dim))
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=100,
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=category,
                                    activation=tf.nn.softmax))
    """
    opti1 = tf.keras.optimizers.Adadelta(lr=0.0001)  # Adadelta
    opti1 = tf.keras.optimizers.Adam(lr=0.0001)  # 使用Adam 移動 0.001  #  內定值 learning_rate=0.001,
    opti1 = tf.keras.optimizers.Nadam(lr=0.0001)
    opti1 = tf.keras.optimizers.SGD(lr=0.0001)  # 梯度下降
    """
    model.compile(optimizer=opti1,
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    history=model.fit(train_x, train_y2,
              epochs=200,
              batch_size=30,
              verbose=0,  # 訓練時顯示的訊息的狀態，0 無顯示、1 進度、2 詳細
              validation_split=0.3  # 如是 0.3，在訓練時會拿 30% 的數據自行驗證數據)
              )
    # 測試
    score = model.evaluate(test_x, test_y2, batch_size=128)
    print("score:", score)
    predict = model.predict(test_x)
    print("Ans:", np.argmax(predict, axis=-1))
    return model, history

model, history_Adadelta=AI_MLP(tf.keras.optimizers.Adadelta(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_Adam=AI_MLP(tf.keras.optimizers.Adam(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_Nadam=AI_MLP(tf.keras.optimizers.Nadam(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)
model, history_SGD=AI_MLP(tf.keras.optimizers.SGD(learning_rate=0.0001),train_x , test_x , train_y , test_y,train_y2 , test_y2)



#
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）

plt.plot(history_Adadelta.history['accuracy'],label="Adadelta 訓練時的正確率")
plt.plot(history_Adam.history['accuracy'],label="Adam 訓練時的正確率")
plt.plot(history_Nadam.history['accuracy'],label="Nadam 訓練時的正確率")
plt.plot(history_SGD.history['accuracy'],label="SGD 訓練時的正確率")
#plt.plot(history.history['loss'],label="訓練時的損失率")
#plt.plot(history.history['val_accuracy'],label="訓練時的驗證正確率")
#plt.plot(history.history['val_loss'],label="訓練時的驗證損失率")

# val_loss: 0.0852 - val_accuracy

plt.title('model accuracy')
plt.ylabel('accuracy , loss, validation accuracy, validation loss')
plt.xlabel('epoch')
plt.legend(loc='lower left')
plt.savefig("MLP曲線.jpg")
plt.show()
