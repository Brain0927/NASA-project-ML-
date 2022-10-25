#!/usr/bin/python
# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn import tree



# Load the diabetes dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from mlxtend.plotting import plot_decision_regions  # pip install mlxtend

# Load the diabetes dataset
#iris = datasets.load_iris()
#iris_X_train , iris_X_test , iris_y_train , iris_y_test = train_test_split(iris.data,iris.target,test_size=0.2)

print("====讀取資料==============")

df = pd.read_excel("NASA(清洗後).xlsx",0) #('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
print(df.columns)

#colX=['Age', 'Sex' , 'ALB', 'ALP',   'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
# sepal length (cm)	sepal width (cm)	petal length (cm)	petal width (cm)	target
# 為了畫出 決策區域， 只能用2 個
colX=[ 'relative_velocity', 'miss_distance']  # <------ 為了畫出 決策區域， 只能用2 個
colY=['hazardous']
# colY2=df['Category'].unique().tolist()  #  文字轉數字

print(df.head())
X=df[colX]
X=np.array(X)
Y=df[colY]
Y=np.array(Y)
Y=Y.ravel()     # 2D　轉1D

print(" X shape",X.shape)
print(" Y shape",Y.shape)



print("====資料拆分==============")
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.02)

print("實際的答案           ：",test_y)


# Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x, train_y.ravel())
prediction = model.predict(test_x)
modelScore=model.score(test_x,test_y)
print("Naive Bayes 預估答案：",prediction," 準確率：",modelScore)

plot_decision_regions(train_x, train_y, clf=model, legend=2)
plt.savefig("決策區域Naive Bayes.jpg")
plt.show()



# 隨機森林

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=2)
rf.fit(train_x, train_y.ravel())
prediction = rf.predict(test_x)
rfScore=rf.score(test_x,test_y)
print("隨機森林 預估答案      ：",prediction," 準確率：",rfScore)
###
from sklearn.tree import export_graphviz
export_graphviz(rf.estimators_[2], out_file='隨機森林1.dot',
                feature_names = colX,
                # class_names =colY2,
                rounded = True, proportion = False,
                precision = 2, filled = True)


plot_decision_regions(train_x, train_y, clf=rf, legend=2)
plt.savefig("決策區域 隨機森林.jpg")
plt.show()




#######
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 5)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names = colX,
                   #class_names=colY2,
                   filled = True,
                   ax = axes[index])

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('隨機森林1.png')
plt.show()




# 決策樹
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x,train_y.ravel())
prediction = clf.predict(test_x)
clfScore=clf.score(test_x,test_y)
print("決策樹 預估答案       ：",prediction," 準確率：",clfScore)

plot_decision_regions(train_x, train_y, clf=clf, legend=2)
plt.savefig("決策區域  決策樹.jpg")
plt.show()

#####
tree.export_graphviz(clf,out_file='決策樹.dot')
# 換成中文的字體
# plt.rcParams['font.新細明體'] = ['SimSun'] # 步驟一（替換sans-serif字型）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False  # 步驟二（解決座標軸負數的負號顯示問題）
plt.rcParams.update({'font.size': 8})
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names = colX,
                   #class_names=colY2,
                   filled=True)
fig.savefig("決策樹1.png")
plt.show()




# KMeans 演算法
kmeans  = KMeans(n_clusters = 3)
kmeans.fit(train_x)
y_predict=kmeans.predict(test_x)
kmeansScore = metrics.accuracy_score(test_y,kmeans.predict(test_x))
print("KMeans 演算法 預估答案：",y_predict," 準確率：",kmeansScore)


plot_decision_regions(train_x, train_y, clf=kmeans, legend=2)
plt.savefig("決策區域_kmeans.jpg")
plt.show()


# KNN 演算法

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, p=1)
knn.fit(train_x, train_y)
knnPredict = knn.predict(test_x)
knnScore=knn.score(test_x, test_y)
print("KNN    演算法 預估答案：",knnPredict," 準確率：",knnScore)

plot_decision_regions(train_x, train_y, clf=knn, legend=2)
plt.savefig("決策區域_knn.jpg")
plt.show()

# 決策樹 演算法
from sklearn import tree
import pydot
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C1.dot')
clfPredict = clf.predict(test_x)
clfScore1=clf.score(test_x, test_y)
print("決策樹 1 演算法 預估答案：",clfPredict," 準確率：",clfScore1)
plot_decision_regions(train_x, train_y, clf=clf, legend=2)
plt.savefig("決策區域_決策樹_gini.jpg")
plt.show()


# 決策樹 演算法

clf = tree.DecisionTreeClassifier( criterion='entropy',splitter='random',max_depth=2)
clf = clf.fit(train_x, train_y)
tree.export_graphviz(clf,out_file='tree-C2.dot')
clfPredict = clf.predict(test_x)
clfScore2=clf.score(test_x, test_y)
print("決策樹 2 演算法 預估答案：",clfPredict," 準確率：",clfScore2)
plot_decision_regions(train_x, train_y, clf=clf, legend=2)
plt.savefig("決策區域_決策樹_entropy.jpg")
plt.show()

#######
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names=colX,
                   #class_names=colY,
                   filled=True)

fig.savefig("decistion_tree.png")
# plt.show()