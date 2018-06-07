# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:41:46 2018

@author: Administrator
"""
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

x_test=[]
y_test = []

for i in range(2):
    print(i)
    file = file = open("../database/image_small_train/00000%d_0"%i,"r")
    for line in file.readlines():
        line =line.strip().split('\t')
        y_test.append(line[0])
        x_test.append([np.float(raw_11) for raw_11 in line[1].split(',')])
#        print (line.strip().split('\t'))
    file.close()
       
x_test = np.array(x_test)

x_train=[]
y_train = []

for i in range(6):
    print(i)
    file = open("../database/54w_256_features/00000%d_0"%i,"r")
    for line in file.readlines():
        line =line.strip().split('\t')
        y_train.append(line[0])
        x_train.append([np.float(raw_11) for raw_11 in line[1].split(',')])
#        print (line.strip().split('\t'))
    file.close()
       
x_train = np.array(x_train)
x_train = np.append(x_train,x_test,axis=0)
from sklearn.cross_validation import train_test_split
##x为数据集的feature熟悉，y为label.
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.3) 

#from sklearn.cluster import DBSCAN
#y_pred = DBSCAN(eps = 0.001, min_samples = 50).fit_predict(heights)

from sklearn.mixture import GMM
GMM_para = GMM(n_components=40,n_iter=200,min_covar = 1e-12).fit(x_train)
y_pred = GMM_para.predict(x_test)

num = {}
print()
for i in y_pred:
    if i in num.keys():
        num[i] +=1 
    else:
        num[i] = 1
print(num,len(num),set(y_pred))
plt.scatter(x_test[:, 110],x_test[:, 220],c=y_pred[:])
plt.show()
test_calculate = {}        
for i in range(len(y_pred)):
    if y_test[i] in test_calculate.keys():
        test_calculate[y_test[i]].append(y_pred[i])
    else:
                    
        test_calculate[y_test[i]] = [y_pred[i]]
file = open('./result.txt',"w")
for k,v in test_calculate.items():
    file.write(k+ ":")
    [file.write(" " + str(v_)) for v_ in v]
    file.write("\n")
file.close()
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
x_train, x_test, y_train, y_test = train_test_split(x_test, y_pred, test_size = 0.1) 
X_embedded = tsne.fit_transform(x_test) #进行数据降维
plt.scatter(X_embedded[:,0],X_embedded[:, 2],c=y_test)
plt.show()
#plt.hist(heights[1], 100, normed=1, facecolor='g', alpha=0.75)

#plt.title('Heights Of feature')
#plt.show()
#from sklearn.cluster import KMeans
#from sklearn import metrics
#print(np.array([heights]).shape)
#kmeans_model = KMeans(n_clusters=256, random_state=128).fit(np.array([heights]).reshape(-1,1))
#labels = kmeans_model.labels_
#print(metrics.calinski_harabaz_score(np.array([heights]).reshape(-1,1), labels) )
#print(metrics.silhouette_score(np.array([heights]).reshape(-1,1), labels, metric='euclidean'))
