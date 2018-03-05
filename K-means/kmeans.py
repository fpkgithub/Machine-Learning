# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 08:55:57 2018

@author: Administrator
"""

import numpy as np


def kmeans(X, k, maxIt):
    numPoints, numDim = X.shape
    dataSet = np.zeros((numPoints, numDim + 1))
    dataSet[:, :-1] = X

    # 随机初始化中心点
    centroids = dataSet[np.random.randint(numPoints, size=k), :]

    # 随机的赋值标签
    centroids[:, -1] = range(k)

    iterations = 0
    oldCentroids = np.zeros((k, numDim + 1))

    # 运行kmeans算法
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print("iteration:\n", iterations)
        print("dataset:\n", dataSet)
        print("centroids:\n", centroids)

        # 保存老中心点用于比较
        oldCentroids = np.copy(centroids)
        iterations += 1

        # 基于中心点更新lable
        updateLabels(dataSet, centroids)

        # 重新更新中心点
        centroids = getCentroids(dataSet, k)
    return dataSet


def shouldStop(oldCentroids, centroids, iterations, maxIt):
    """
    当迭代次数大于最大迭代次数时候停止
    当中心点不在更新的时候迭代停止
    """
    if iterations > maxIt:
        return True
    return np.array_equal(centroids, oldCentroids)
    # return False


def updateLabels(dataSet, centroids):
    """
    根据中心点进行更新lables
    """
    numPoints, numDim = centroids.shape
    distanceMatric = []
    for i in range(0, numPoints):
        distanceMatric.append(np.linalg.norm(dataSet[:, :-1] - np.atleast_2d(centroids[i, :-1]), axis=1))
    distanceMatric = np.atleast_2d(distanceMatric)
    labelIndex = np.argmin(distanceMatric, axis=0)
    dataSet[:, -1] = labelIndex.T


def getCentroids(dataSet, k):
    centroids = []
    for i in range(k):
        mean = np.mean(dataSet[dataSet[:, -1] == i, :-1], axis=0)
        centroids.append(np.hstack((mean, [i])))

    return np.atleast_2d(centroids)


import h5py


def loadDataSet():
    '''
    加载图片数据集
    '''
    pictureSet = [];
    pictureClasses = []
    with h5py.File(r'F:\PyCharm\Room\datamining\id3\algorithm\data\kmean\Land.h5') as h5f:
        forest = h5f['forest'][:]
        urban = h5f['urban'][:]
        pictureSet = np.vstack((forest, urban))
        pictureClasses = [0] * 50 + [1] * 50
    return pictureSet, pictureClasses


if __name__ == '__main__':
    # x1 = np.array([1, 1])
    # x2 = np.array([2, 1])
    # x3 = np.array([4, 3])
    # x4 = np.array([5, 4])
    # testX = np.vstack((x1, x2, x3, x4))
    testX, testY = loadDataSet()
    result = kmeans(testX, 2, 1000)
    print(np.sum(np.equal(testY, result[:, -1])) / 100.)
