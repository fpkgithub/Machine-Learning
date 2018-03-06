# SVM

##简单概念

先介绍一些简单的基本概念

**分隔超平面**：将数据集分割开来的直线叫做分隔超平面。

**超平面**：如果数据集是N维的，那么就需要N-1维的某对象来对数据进行分割。该对象叫做超平面，也就是分类的决策边界。

**间隔**：

> 一个点到分割面的距离，称为点相对于分割面的距离。
>
> 数据集中所有的点到分割面的最小间隔的2倍，称为分类器或数据集的间隔。

**最大间隔**：SVM分类器是要找最大的数据集间隔。

**支持向量**：坐落在数据边际的两边超平面上的点被称为支持向量

 ## 公式

转载：[支持向量机SVM通俗理解（python代码实现）](http://blog.csdn.net/csqazwsxedc/article/details/71513197)

这是第三次来“复习”SVM了，第一次是使用SVM包，调用包并尝试调节参数。听闻了“流弊”SVM的算法。第二次学习理论，看了李航的《统计学习方法》以及网上的博客。看完后感觉，满满的公式。。。记不住啊。第三次，也就是这次通过python代码手动来实现SVM，才让我突然对SVM不有畏惧感。希望这里我能通过简单粗暴的文字，能让读者理解到底什么是SVM，这货的算法思想是怎么样的。看之前千万不要畏惧，说到底就是个算法，每天啃一点，总能啃完它，慢慢来还可以加深印象。 
SVM是用来解决分类问题的，如果解决两个变量的分类问题，可以理解成用一条直线把点给分开，完成分类。如下： 
![这里写图片描述](http://img.blog.csdn.net/20140829135959290) 
上面这些点很明显不一样，我们从中间画一条直线就可以用来分割这些点，但是什么样的直线才是最好的呢？通俗的说，就是一条直线“最能”分割这些点，也就是上图中的直线。他是最好的一条直线，使所有的点都“尽量”远离中间那条直线。总得的来说，SVM就是为了找出一条分割的效果最好的直线。怎么样找出这条直线，就变成了一个数学问题，通过数学一步一步的推导，最后转化成程序。这里举例是二个特征的分类问题，如果有三个特征，分类线就变成了分类平面，多个特征的话就变成了超平面。从这个角度出发去看待SVM，会比较轻松。

数学解决方法大致如下： 
目的是求最大分隔平面，也就是选取靠近平面最近的点，使这些点到分隔平面的距离W最大，是一个典型的凸二次规划问题。 
![这里写图片描述](http://img.blog.csdn.net/20170510234220486?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3NxYXp3c3hlZGM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
但是上面需要求解两个参数w和b；于是为求解这个问题，把二次规划问题转换为对偶问题 
![这里写图片描述](http://img.blog.csdn.net/20170510234554210?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3NxYXp3c3hlZGM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
这样就只需求一个参数a了，通过SMO算法求出a后，再计算出b 
![这里写图片描述](http://img.blog.csdn.net/20170510234723982?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3NxYXp3c3hlZGM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
最后通过f(x)用做预测。

详细的数学推导，请看下面两个博客以及《统计学习方法》，这两位博主其实已经讲解的非常详细了。 
<http://blog.csdn.net/zouxy09/article/details/17291543> 
<http://blog.csdn.net/v_july_v/article/details/7624837> 
《统计学习方法》这本书里面全是数学公式，非常“课本”，建议先看博客，有个大概印象再去看“课本”，跟着“课本”一步一步的推导。最后用python代码实现一遍，应该就可以拿下SVM了。

python代码实现可以加深对那些数学推导公式的印象，看公式的时候，可能会想，这些推导好复杂，都有些什么用啊，结果写代码的时候会发现，原来最后都用在代码里。所以写代码可以加深对SVM的理解。 
下面是SVM的python代码实现，我做了详细的注释，刚开始看代码也会觉得好长好复杂，慢慢看后发现，代码就是照着SVM的数学推导，把最后的公式推导转化为代码和程序的逻辑，代码本身并不复杂。

## 代码

```python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/6 9:35
# @Author  : Boy

from numpy import *


def loadDataSet(filename):  # 读取数据
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        str = lineArr[0]
        strint = float(str)
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat  # 返回数据特征和数据类别


def selectJrand(i, m):  # 在0-m中随机选择一个不是i的整数
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 保证a在L和H范围内（L <= a <= H）
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup):  # 核函数，输入参数,X:支持向量的特征树；A：某一行特征数据；kTup：('lin',k1)核函数的类型和参数
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性函数
        K = X * A.T
    elif kTup[0] == 'rbf':  # 径向基函数(radial bias function)
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # 返回生成的结果
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


# 定义类，方便存储数据
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # 存储各类参数
        self.X = dataMatIn  # 数据特征
        self.labelMat = classLabels  # 数据类别
        self.C = C  # 软间隔参数C，参数越大，非线性拟合能力越强
        self.tol = toler  # 停止阀值
        self.m = shape(dataMatIn)[0]  # 数据行数
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0  # 初始设为0
        self.eCache = mat(zeros((self.m, 2)))  # 缓存
        self.K = mat(zeros((self.m, self.m)))  # 核函数的计算结果
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk(oS, k):  # 计算Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


# 随机选取aj，并返回其E值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 返回矩阵中的非零位置的行数
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):  # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):  # 更新os数据
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 首先检验ai是否满足KKT条件，如果不满足，随机选择aj进行优化，更新ai,aj,b值
def innerL(i, oS):  # 输入参数i和所有参数数据
    Ei = calcEk(oS, i)  # 计算E值
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):  # 检验这行数据是否符合KKT条件 参考《统计学习方法》p128公式7.111-113
        j, Ej = selectJ(i, oS, Ei)  # 随机选取aj，并返回其E值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):  # 以下代码的公式参考《统计学习方法》p126
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # 参考《统计学习方法》p127公式7.107
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # 参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  # 参考《统计学习方法》p127公式7.108
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < oS.tol):  # alpha变化大小阀值（自己设定）
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # 参考《统计学习方法》p127公式7.109
        updateEk(oS, i)  # 更新数据
        # 以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# SMO函数，用于快速求解出alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # 输入参数：数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    print("---------------SMO函数-----------------")
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        print("------------------迭代第%d次--------------------"%(iter+1))
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):  # 遍历所有数据
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:  # 遍历非边界的数据
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def testRbf(data_train, data_test):
    dataArr, labelArr = loadDataSet(data_train)  # 读取训练数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', 1.3))  # 通过SMO算法得到b和alpha
    # mat函数可以将目标函数的类型转换为矩阵
    datMat = mat(dataArr)
    # 矩阵转秩
    labelMat = mat(labelArr).transpose()
    # 选取不为0数据的行数（也就是支持向量）
    svInd = nonzero(alphas)[0]
    sVs = datMat[svInd]  # 支持向量的特征数据
    labelSV = labelMat[svInd]  # 支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0])  # 打印出共有多少的支持向量
    #查看矩阵或者数组的维数
    m, n = shape(datMat)  # 训练数据的行列数
    errorCount = 0
    for i in range(m):
        # 将支持向量转化为核函数
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))
        # 这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    # 打印出错误率
    print("----------------------打印训练出错误率---------------------")
    print("the training error rate is: %f" % (float(errorCount) / m))
    # 读取测试数据
    dataArr_test, labelArr_test = loadDataSet(data_test)
    errorCount_test = 0
    datMat_test = mat(dataArr_test)
    labelMat = mat(labelArr_test).transpose()
    m, n = shape(datMat_test)
    for i in range(m):  # 在测试数据上检验错误率
        kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr_test[i]):
            errorCount_test += 1
    print("----------------------打印测试出错误率---------------------")
    print("the test error rate is: %f" % (float(errorCount_test) / m))


def write_results(listResult, dataset, k):
    with open('kmeans2-result.txt', 'a') as file:
        for kind in range(k):
            file.write("CLASSINFO:%d--%d\n" % (kind + 1,len(listResult[kind])))
            for j in listResult[kind]:
                file.write('%d ' % j)
            file.write('\n')
        file.write('\n\n')
        file.close()

# 主程序
def main():
    filename_traindata = '../data/Svm/train_data.txt'
    filename_testdata = '../data/Svm/test_data.txt'
    testRbf(filename_traindata, filename_testdata)


if __name__ == '__main__':
    main()
```

样例数据如下： 
![这里写图片描述](http://img.blog.csdn.net/20170511023030158?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3NxYXp3c3hlZGM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

##数据集 

训练数据：train_data.txt

测试数据：test_data.txt











