#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ ="FU-LONG,SHI"
#參考資料 https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects

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
import 標準化
from sklearn.cluster import KMeans
from sklearn import metrics
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'

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

train_x, test_x, train_y, test_y,scaler=標準化.ML_read_dataframe_標準化("NASA(清洗後).xlsx", colX, col_target)
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



# Bayes


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y.ravel())
prediction = model.predict(test_x)
modelScore=model.score(test_x,test_y)
print("Naive Bayes 預估答案：",prediction," 準確率：",modelScore)


# 隨機森林

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=3,
                             random_state=2)
rf.fit(train_x, train_y2)
prediction = rf.predict(test_x)
rfScore=rf.score(test_x,test_y2)
print("隨機森林 預估答案      ：",prediction," 準確率：",rfScore)
###
from sklearn.tree import export_graphviz
export_graphviz(rf.estimators_[2], out_file='隨機森林1.dot',
                feature_names = colX,
                # class_names =colY2,
                rounded = True, proportion = False,
                precision = 2, filled = True)


#######
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 5 )
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = colX,
                   #class_names=colY2,
                   filled = True,
                   ax = axes[index],
                   fontsize = 8)
    axes[index].set_title('Estimator: ' + str(index), fontsize =20)

fig.savefig('隨機森林1.png')
plt.show()


# 決策樹
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x,train_y2)
prediction = clf.predict(test_x)
clfScore=clf.score(test_x,test_y2)
print("決策樹 預估答案：",prediction," 準確率：",clfScore)


#####
tree.export_graphviz(clf,out_file='決策樹.dot')
# 換成中文的字體
# plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'

plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
plt.rcParams.update({'font.size': 20})
fig = plt.figure()
tree.plot_tree(clf,
                   feature_names = colX,
                   #class_names=colY2,
                   filled=True)
fig.savefig("決策樹1.png")
plt.show()




# KMeans 演算法
kmeans  = KMeans() #演算法
kmeans.fit(train_x)  #訓練
y_predict=kmeans.predict(test_x)  #預測
kmeansScore = metrics.accuracy_score(test_y,kmeans.predict(test_x))  #分數
kmeanshomogeneity_score= metrics.homogeneity_score(test_y,kmeans.predict(test_x))  #分數
print("KMeans 演算法 預估答案：",y_predict," 準確率：",kmeansScore)
print("KMeans 演算法 預估答案：",y_predict," 修正後準確率：",kmeanshomogeneity_score)






# KNN 演算法

from sklearn.neighbors import KNeighborsClassifier #knn演算法
knn = KNeighborsClassifier(n_neighbors=3, p=1)  #初始化
knn.fit(train_x, train_y) #訓練
knnPredict = knn.predict(test_x) #預測
knnScore=knn.score(test_x, test_y)   #分數
print("KNN    演算法 預估答案：",knnPredict," 準確率：",knnScore)

# 決策樹 演算法
from sklearn import tree   #決策樹 演算法
import pydot
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C1.dot')
clfPredict = clf.predict(test_x)
clfScore1=clf.score(test_x, test_y)
print("決策樹 1 gini 預估答案：",clfPredict," 準確率：",clfScore1)



# 決策樹 演算法
clf = tree.DecisionTreeClassifier( criterion='entropy',splitter='random',max_depth=2)
clf = clf.fit(train_x, train_y2)
tree.export_graphviz(clf,out_file='tree-C2.dot')
clfPredict = clf.predict(test_x)
clfScore2=clf.score(test_x, test_y2)
print("決策樹 2 entropy 預估答案：",clfPredict," 準確率：",clfScore2)


#######
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names=colX,
                   #class_names=colY,
                   filled=True)

fig.savefig("decistion_tree.png")
plt.show()
