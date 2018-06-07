# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:48:47 2018

@author: Administrator
"""
import matplotlib.pyplot as plt  
import numpy as np

src_points = [0.0, 0.128612995148, 0.236073002219, 0.316282004118, 0.382878988981, 0.441266000271, 0.490464001894, 1.0]

dst_points = [0.0, 0.40000000596, 0.5, 0.600000023842, 0.699999988079, 0.800000011921, 0.899999976158, 1.0]

def cos_distance_vec(vecA, vecB):
    return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
def normalize(score):
    srcLen = len(src_points)
    if score <= src_points[0]:
        return 0.0
    elif score >= src_points[srcLen - 1]:
        return 1.0

    result = 0.0

    for i in range(1, srcLen):
        if score < src_points[i]:
            result = dst_points[i - 1] + (score - src_points[i - 1]) * (dst_points[i] - dst_points[i - 1]) / (
                src_points[i] - src_points[i - 1])
            break

    return result
x_test=[]
y_test = []

for i in range(2):
    file = file = open("./image_small_train/00000%d_0"%i,"r")
    for line in file.readlines():
        line =line.strip().split('\t')
        if str(line[0]) in ["79047","79021"]:
            
            y_test.append(line[0])
            x_test.append([np.float(raw_11) for raw_11 in line[1].split(',')])
#        print (line.strip().split('\t'))
    file.close()
x_test = np.array(x_test)
x_mean = np.mean(x_test,axis = 0)
for i in range(len(x_test)):
    for j in range(i+1,len(x_test)):
        print(cos_distance_vec(x_test[i],x_test[j]))
for num in x_test:
    print(cos_distance_vec(num,x_mean))
#x = [i for i in range(26)]
#plt.plot(x,x_test[0][::10],'r',x,x_test[1][::10],'b',x,x_test[2][::10],'g')
#plt.show()