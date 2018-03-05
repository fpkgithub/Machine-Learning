- Kmeans聚类算法及其 Python实现
  - [关于聚类](http://blog.csdn.net/fzch_struggling/article/details/45009097#%E5%85%B3%E4%BA%8E%E8%81%9A%E7%B1%BB)
  - [基本思想](http://blog.csdn.net/fzch_struggling/article/details/45009097#%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3)
  - [初始质心的选择](http://blog.csdn.net/fzch_struggling/article/details/45009097#%E5%88%9D%E5%A7%8B%E8%B4%A8%E5%BF%83%E7%9A%84%E9%80%89%E6%8B%A9)
  - [算法实验](http://blog.csdn.net/fzch_struggling/article/details/45009097#%E7%AE%97%E6%B3%95%E5%AE%9E%E9%AA%8C)
  - [Python实现](http://blog.csdn.net/fzch_struggling/article/details/45009097#python%E5%AE%9E%E7%8E%B0)

> **本节内容：**本节内容是根据上学期所上的模式识别课程的作业整理而来，第一道题目是Kmeans聚类算法，数据集是Iris(鸢尾花的数据集)，分类数k是3，数据维数是4。

------

Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

## 关于聚类

​    聚类算法是这样的一种算法：给定样本数据Sample，要求将样本Sample中相似的数据聚到一类。有了这个认识之后，就应该了解了聚类算法要干什么了吧。说白了，就是归类。 
​    首先，我们需要考虑的是，如何衡量数据之间的相似程度？比如说，有一群说不同语言的人，我们一般是根据他们的方言来聚类的（当然，你也可以指定以身高来聚类）。这里，语言的相似性（或者身高）就成了我们衡量相似的量度了。在考虑存在海量数据，如微博上各种用户的关系网，如何根据用户的关注和被关注来聚类，给用户推荐他们感兴趣的用户？这就是聚类算法研究的内容之一了。 
​    Kmeans就是这样的聚类算法中比较简单的算法，给定数据样本集Sample和应该划分的类数K，对样本数据Sample进行聚类，最终形成K个cluster，其相似的度量是某条数据i与中心点的”距离”(这里所说的距离，不止于二维)。

## 基本思想

KMeans算法的**基本思想**是初始随机给定K个簇中心，按照最邻近原则把待分类样本点分到各个簇。然后按平均法重新计算各个簇的质心，从而确定新的簇心。一直迭代，直到簇心的移动距离小于某个给定的值。

- **基本步骤** 
  K-Means聚类算法主要分为三个步骤： 
  1，初始化k个聚类中心。 
  2，计算出每个对象跟这k个中心的距离（相似度计算，这个下面会提到），假如x这个对象跟y这个中心的距离最小（相似度最大），那么x属于y这个中心。这一步就可以得到初步的k个聚类 。 
  3，在第二步得到的每个聚类分别计算出新的聚类中心，和旧的中心比对，假如不相同，则继续第2步，直到新旧两个中心相同，说明聚类不可变，已经成功 。
- **复杂度分析** 
  时间复杂度：O(tKmn)，其中，t为迭代次数，K为簇的数目，m为记录数，n为维数 
  空间复杂度：O((m+K)n)，其中，K为簇的数目，m为记录数，n为维数

------

## 初始质心的选择

​    选择适当的初始质心是基本kmeans算法的关键步骤。常见的方法是随机的选取初始质心，但是这样簇的质量常常很差。处理选取初始质心问题的一种**常用技术**是：多次运行，每次使用一组不同的随机初始质心，然后选取具有最小SSE（误差的平方和）的簇集。这种策略简单，但是效果可能不好，这取决于数据集和寻找的簇的个数。 

​     第二种有效的方法是，取一个样本，并使用层次聚类技术对它聚类。从层次聚类中提取K个簇，并用这些簇的质心作为初始质心。该方法通常很有效，但仅对下列情况有效： 
​        （1）样本相对较小，例如数百到数千（层次聚类开销较大）； 
​        （2）K相对于样本大小较小

​    第三种选择初始质心的方法，随机地选择第一个点，或取所有点的质心作为第一个点。然后，对于每个后继初始质心，选择离已经选取过的初始质心最远的点。使用这种方法，确保了选择的初始质心不仅是随机的，而且是散开的。但是，这种方法可能选中离群点。此外，求离当前初始质心集最远的点开销也非常大。为了克服这个问题，通常该方法用于点样本。由于离群点很少（多了就不是离群点了），它们多半不会在随机样本中出现。计算量也大幅减少。 

​    第四种方法是**使用canopy算法进行初始划分**。基于Canopy Method的聚类算法将聚类过程分为两个阶段： 
   **Stage1**：聚类最耗费计算的地方是计算对象相似性的时候，Canopy Method在第一阶段选择简单、计算代价较低的方法计算对象相似性，将相似的对象放在一个子集中，这个子集被叫做Canopy ，通过一系列计算得到若干Canopy，Canopy之间可以是重叠的，但不会存在某个对象不属于任何Canopy的情况，可以把这一阶段看做数据预处理。 
  **Stage2**：在各个Canopy 内使用传统的聚类方法(如K-means)，不属于同一Canopy 的对象之间不进行相似性计算。从这个方法起码可以看出两点好处：首先，Canopy 不要太大且Canopy 之间重叠的不要太多的话会大大减少后续需要计算相似性的对象的个数；其次，类似于K-means这样的聚类方法是需要人为指出K的值的，通过Stage1得到的Canopy 个数完全可以作为这个K值，一定程度上减少了选择K的盲目性。

------

## 算法实验

- **任务** 
  在给定的Iris.txt样本文件中，用K-means聚类算法将150个4维样本数据分成3类
- **数据集(Iris.txt)**

以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，是在数据挖掘、数据分类中非常常用的测试集、训练集

三类分别为:setosa, versicolor, virginica

数据包含4个独立的属性,这些属性变量测量植物的花朵,比如萼片长度, 萼片宽度,花瓣长度, 花瓣宽度.

| **Sepal length** | **Sepal width** | **Petal length** | **Petal width** | **Species**     |
| :--------------- | --------------- | ---------------- | --------------- | --------------- |
| 5.1              | 3.5             | 1.4              | 0.2             | *I. setosa*     |
| 4.9              | 3               | 1.4              | 0.2             | *I. setosa*     |
| 4.7              | 3.2             | 1.3              | 0.2             | *I. setosa*     |
| 4.6              | 3.1             | 1.5              | 0.2             | *I. setosa*     |
| 5                | 3.6             | 1.4              | 0.3             | *I. setosa*     |
| 5.4              | 3.9             | 1.7              | 0.4             | *I. setosa*     |
| 4.6              | 3.4             | 1.4              | 0.3             | *I. setosa*     |
| 5                | 3.4             | 1.5              | 0.2             | *I. setosa*     |
| 4.4              | 2.9             | 1.4              | 0.2             | *I. setosa*     |
| 4.9              | 3.1             | 1.5              | 0.1             | *I. setosa*     |
| 5.4              | 3.7             | 1.5              | 0.2             | *I. setosa*     |
| 4.8              | 3.4             | 1.6              | 0.2             | *I. setosa*     |
| 4.8              | 3               | 1.4              | 0.1             | *I. setosa*     |
| 4.3              | 3               | 1.1              | 0.1             | *I. setosa*     |
| 5.8              | 4               | 1.2              | 0.2             | *I. setosa*     |
| 5.7              | 4.4             | 1.5              | 0.4             | *I. setosa*     |
| 5.4              | 3.9             | 1.3              | 0.4             | *I. setosa*     |
| 5.1              | 3.5             | 1.4              | 0.3             | *I. setosa*     |
| 5.7              | 3.8             | 1.7              | 0.3             | *I. setosa*     |
| 5.1              | 3.8             | 1.5              | 0.3             | *I. setosa*     |
| 5.4              | 3.4             | 1.7              | 0.2             | *I. setosa*     |
| 5.1              | 3.7             | 1.5              | 0.4             | *I. setosa*     |
| 4.6              | 3.6             | 1                | 0.2             | *I. setosa*     |
| 5.1              | 3.3             | 1.7              | 0.5             | *I. setosa*     |
| 4.8              | 3.4             | 1.9              | 0.2             | *I. setosa*     |
| 5                | 3               | 1.6              | 0.2             | *I. setosa*     |
| 5                | 3.4             | 1.6              | 0.4             | *I. setosa*     |
| 5.2              | 3.5             | 1.5              | 0.2             | *I. setosa*     |
| 5.2              | 3.4             | 1.4              | 0.2             | *I. setosa*     |
| 4.7              | 3.2             | 1.6              | 0.2             | *I. setosa*     |
| 4.8              | 3.1             | 1.6              | 0.2             | *I. setosa*     |
| 5.4              | 3.4             | 1.5              | 0.4             | *I. setosa*     |
| 5.2              | 4.1             | 1.5              | 0.1             | *I. setosa*     |
| 5.5              | 4.2             | 1.4              | 0.2             | *I. setosa*     |
| 4.9              | 3.1             | 1.5              | 0.2             | *I. setosa*     |
| 5                | 3.2             | 1.2              | 0.2             | *I. setosa*     |
| 5.5              | 3.5             | 1.3              | 0.2             | *I. setosa*     |
| 4.9              | 3.6             | 1.4              | 0.1             | *I. setosa*     |
| 4.4              | 3               | 1.3              | 0.2             | *I. setosa*     |
| 5.1              | 3.4             | 1.5              | 0.2             | *I. setosa*     |
| 5                | 3.5             | 1.3              | 0.3             | *I. setosa*     |
| 4.5              | 2.3             | 1.3              | 0.3             | *I. setosa*     |
| 4.4              | 3.2             | 1.3              | 0.2             | *I. setosa*     |
| 5                | 3.5             | 1.6              | 0.6             | *I. setosa*     |
| 5.1              | 3.8             | 1.9              | 0.4             | *I. setosa*     |
| 4.8              | 3               | 1.4              | 0.3             | *I. setosa*     |
| 5.1              | 3.8             | 1.6              | 0.2             | *I. setosa*     |
| 4.6              | 3.2             | 1.4              | 0.2             | *I. setosa*     |
| 5.3              | 3.7             | 1.5              | 0.2             | *I. setosa*     |
| 5                | 3.3             | 1.4              | 0.2             | *I. setosa*     |
| 7                | 3.2             | 4.7              | 1.4             | *I. versicolor* |
| 6.4              | 3.2             | 4.5              | 1.5             | *I. versicolor* |
| 6.9              | 3.1             | 4.9              | 1.5             | *I. versicolor* |
| 5.5              | 2.3             | 4                | 1.3             | *I. versicolor* |
| 6.5              | 2.8             | 4.6              | 1.5             | *I. versicolor* |
| 5.7              | 2.8             | 4.5              | 1.3             | *I. versicolor* |
| 6.3              | 3.3             | 4.7              | 1.6             | *I. versicolor* |
| 4.9              | 2.4             | 3.3              | 1               | *I. versicolor* |
| 6.6              | 2.9             | 4.6              | 1.3             | *I. versicolor* |
| 5.2              | 2.7             | 3.9              | 1.4             | *I. versicolor* |
| 5                | 2               | 3.5              | 1               | *I. versicolor* |
| 5.9              | 3               | 4.2              | 1.5             | *I. versicolor* |
| 6                | 2.2             | 4                | 1               | *I. versicolor* |
| 6.1              | 2.9             | 4.7              | 1.4             | *I. versicolor* |
| 5.6              | 2.9             | 3.6              | 1.3             | *I. versicolor* |
| 6.7              | 3.1             | 4.4              | 1.4             | *I. versicolor* |
| 5.6              | 3               | 4.5              | 1.5             | *I. versicolor* |
| 5.8              | 2.7             | 4.1              | 1               | *I. versicolor* |
| 6.2              | 2.2             | 4.5              | 1.5             | *I. versicolor* |
| 5.6              | 2.5             | 3.9              | 1.1             | *I. versicolor* |
| 5.9              | 3.2             | 4.8              | 1.8             | *I. versicolor* |
| 6.1              | 2.8             | 4                | 1.3             | *I. versicolor* |
| 6.3              | 2.5             | 4.9              | 1.5             | *I. versicolor* |
| 6.1              | 2.8             | 4.7              | 1.2             | *I. versicolor* |
| 6.4              | 2.9             | 4.3              | 1.3             | *I. versicolor* |
| 6.6              | 3               | 4.4              | 1.4             | *I. versicolor* |
| 6.8              | 2.8             | 4.8              | 1.4             | *I. versicolor* |
| 6.7              | 3               | 5                | 1.7             | *I. versicolor* |
| 6                | 2.9             | 4.5              | 1.5             | *I. versicolor* |
| 5.7              | 2.6             | 3.5              | 1               | *I. versicolor* |
| 5.5              | 2.4             | 3.8              | 1.1             | *I. versicolor* |
| 5.5              | 2.4             | 3.7              | 1               | *I. versicolor* |
| 5.8              | 2.7             | 3.9              | 1.2             | *I. versicolor* |
| 6                | 2.7             | 5.1              | 1.6             | *I. versicolor* |
| 5.4              | 3               | 4.5              | 1.5             | *I. versicolor* |
| 6                | 3.4             | 4.5              | 1.6             | *I. versicolor* |
| 6.7              | 3.1             | 4.7              | 1.5             | *I. versicolor* |
| 6.3              | 2.3             | 4.4              | 1.3             | *I. versicolor* |
| 5.6              | 3               | 4.1              | 1.3             | *I. versicolor* |
| 5.5              | 2.5             | 4                | 1.3             | *I. versicolor* |
| 5.5              | 2.6             | 4.4              | 1.2             | *I. versicolor* |
| 6.1              | 3               | 4.6              | 1.4             | *I. versicolor* |
| 5.8              | 2.6             | 4                | 1.2             | *I. versicolor* |
| 5                | 2.3             | 3.3              | 1               | *I. versicolor* |
| 5.6              | 2.7             | 4.2              | 1.3             | *I. versicolor* |
| 5.7              | 3               | 4.2              | 1.2             | *I. versicolor* |
| 5.7              | 2.9             | 4.2              | 1.3             | *I. versicolor* |
| 6.2              | 2.9             | 4.3              | 1.3             | *I. versicolor* |
| 5.1              | 2.5             | 3                | 1.1             | *I. versicolor* |
| 5.7              | 2.8             | 4.1              | 1.3             | *I. versicolor* |
| 6.3              | 3.3             | 6                | 2.5             | *I. virginica*  |
| 5.8              | 2.7             | 5.1              | 1.9             | *I. virginica*  |
| 7.1              | 3               | 5.9              | 2.1             | *I. virginica*  |
| 6.3              | 2.9             | 5.6              | 1.8             | *I. virginica*  |
| 6.5              | 3               | 5.8              | 2.2             | *I. virginica*  |
| 7.6              | 3               | 6.6              | 2.1             | *I. virginica*  |
| 4.9              | 2.5             | 4.5              | 1.7             | *I. virginica*  |
| 7.3              | 2.9             | 6.3              | 1.8             | *I. virginica*  |
| 6.7              | 2.5             | 5.8              | 1.8             | *I. virginica*  |
| 7.2              | 3.6             | 6.1              | 2.5             | *I. virginica*  |
| 6.5              | 3.2             | 5.1              | 2               | *I. virginica*  |
| 6.4              | 2.7             | 5.3              | 1.9             | *I. virginica*  |
| 6.8              | 3               | 5.5              | 2.1             | *I. virginica*  |
| 5.7              | 2.5             | 5                | 2               | *I. virginica*  |
| 5.8              | 2.8             | 5.1              | 2.4             | *I. virginica*  |
| 6.4              | 3.2             | 5.3              | 2.3             | *I. virginica*  |
| 6.5              | 3               | 5.5              | 1.8             | *I. virginica*  |
| 7.7              | 3.8             | 6.7              | 2.2             | *I. virginica*  |
| 7.7              | 2.6             | 6.9              | 2.3             | *I. virginica*  |
| 6                | 2.2             | 5                | 1.5             | *I. virginica*  |
| 6.9              | 3.2             | 5.7              | 2.3             | *I. virginica*  |
| 5.6              | 2.8             | 4.9              | 2               | *I. virginica*  |
| 7.7              | 2.8             | 6.7              | 2               | *I. virginica*  |
| 6.3              | 2.7             | 4.9              | 1.8             | *I. virginica*  |
| 6.7              | 3.3             | 5.7              | 2.1             | *I. virginica*  |
| 7.2              | 3.2             | 6                | 1.8             | *I. virginica*  |
| 6.2              | 2.8             | 4.8              | 1.8             | *I. virginica*  |
| 6.1              | 3               | 4.9              | 1.8             | *I. virginica*  |
| 6.4              | 2.8             | 5.6              | 2.1             | *I. virginica*  |
| 7.2              | 3               | 5.8              | 1.6             | *I. virginica*  |
| 7.4              | 2.8             | 6.1              | 1.9             | *I. virginica*  |
| 7.9              | 3.8             | 6.4              | 2               | *I. virginica*  |
| 6.4              | 2.8             | 5.6              | 2.2             | *I. virginica*  |
| 6.3              | 2.8             | 5.1              | 1.5             | *I. virginica*  |
| 6.1              | 2.6             | 5.6              | 1.4             | *I. virginica*  |
| 6.3              | 3.4             | 5.6              | 2.4             | *I. virginica*  |
| 6.4              | 3.1             | 5.5              | 1.8             | *I. virginica*  |
| 6                | 3               | 4.8              | 1.8             | *I. virginica*  |
| 6.9              | 3.1             | 5.4              | 2.1             | *I. virginica*  |
| 6.7              | 3.1             | 5.6              | 2.4             | *I. virginica*  |
| 6.9              | 3.1             | 5.1              | 2.3             | *I. virginica*  |
| 5.8              | 2.7             | 5.1              | 1.9             | *I. virginica*  |
| 6.8              | 3.2             | 5.9              | 2.3             | *I. virginica*  |
| 6.7              | 3.3             | 5.7              | 2.5             | *I. virginica*  |
| 6.7              | 3               | 5.2              | 2.3             | *I. virginica*  |
| 6.3              | 2.5             | 5                | 1.9             | *I. virginica*  |
| 6.5              | 3               | 5.2              | 2               | *I. virginica*  |
| 6.2              | 3.4             | 5.4              | 2.3             | *I. virginica*  |
| 5.9              | 3               | 5.1              | 1.8             | *I. virginica*  |

## Python实现

- **算法流程**
  - 第一步，将文件中的数据读入到dataset列表中，通过len(dataset[0])来获取数据维数，在测试样例中是四维
  - 第二步，产生聚类的初始位置。首先扫描数据，获取每一维数据分量中的最大值和最小值，然后在这个区间上随机产生一个值，循环k次(k为所分的类别),这样就产生了聚类初始中心（k个）
  - 第三步，按照最短距离（欧式距离）原则将所有样本分配到k个聚类中心中的某一个，这步操作的结果是产生列表assigments，可以通过Python中的zip函数整合成字典。注意到原始聚类中心可能不在样本中，因此可能出现分配的结果出现某一个聚类中心点集合为空，此时需要结束，提示“随机数产生错误，需要重新运行”，以产生合适的初始中心。
  - 第四步，计算各个聚类中心的新向量，更新距离，即每一类中每一维均值向量。然后再进行分配，比较前后两个聚类中心向量是否相等，若不相等则进行循环，否则终止循环，进入下一步。
  - 最后，将结果输出到文件和屏幕中
- **代码如下**

```python
# coding=gbk
#python edition: Python3.4.1,2014,9,24
from collections import defaultdict
from random import uniform
from math import sqrt

def read_points():
    dataset=[]
    with open('Iris.txt','r') as file:
        for line in file:
            if line =='\n':
                continue
            dataset.append(list(map(float,line.split(' '))))
        file.close() 
        return  dataset

def write_results(listResult,dataset,k):
    with open('result.txt','a') as file:
        for kind in range(k):
              file.write( "CLASSINFO:%d\n"%(kind+1) )
              for j in listResult[kind]:
                 file.write('%d\n'%j)
              file.write('\n')
        file.write('\n\n')
        file.close()

def point_avg(points):
    dimensions=len(points[0])
    new_center=[]
    for dimension in range(dimensions):
        sum=0
        for p in points:
            sum+=p[dimension]
        new_center.append(float("%.8f"%(sum/float(len(points)))))
    return new_center

def update_centers(data_set ,assignments,k):
    new_means = defaultdict(list)
    centers = []
    for assignment ,point in zip(assignments , data_set):
        new_means[assignment].append(point)
    for i in range(k):
        points=new_means[i]
        centers.append(point_avg(points))
    return centers

def assign_points(data_points,centers):
    assignments=[]
    for point in data_points:
        shortest=float('inf')
        shortest_index = 0
        for i in range(len(centers)):
            value=distance(point,centers[i])
            if value<shortest:
                shortest=value
                shortest_index=i
        assignments.append(shortest_index)
    if len(set(assignments))<len(centers) :
           print("\n--!!!产生随机数错误，请重新运行程序！!!!--\n")
           exit()
    return assignments

def distance(a,b):
    dimention=len(a)
    sum=0
    for i in range(dimention):
        sq=(a[i]-b[i])**2
        sum+=sq
    return sqrt(sum)

def generate_k(data_set,k):
    centers=[]
    dimentions=len(data_set[0])
    min_max=defaultdict(int)
    for point in data_set:
        for i in range(dimentions):
            value=point[i]
            min_key='min_%d'%i
            max_key='max_%d'%i
            if min_key not in min_max or value<min_max[min_key]:
                min_max[min_key]=value
            if max_key not in min_max or value>min_max[max_key]:
                min_max[max_key]=value
    for j in range(k):
        rand_point=[]
        for i in range(dimentions):
            min_val=min_max['min_%d'%i]
            max_val=min_max['max_%d'%i]
            tmp=float("%.8f"%(uniform(min_val,max_val)))
            rand_point.append(tmp)
        centers.append(rand_point)
    return centers

def k_means(dataset,k):
    k_points=generate_k(dataset,k)
    assignments=assign_points(dataset,k_points)
    old_assignments=None
    while assignments !=old_assignments:
        new_centers=update_centers(dataset,assignments,k)
        old_assignments=assignments
        assignments=assign_points(dataset,new_centers)
    result=list(zip(assignments,dataset))
    print('\n\n---------------------------------分类结果---------------------------------------\n\n')
    for out in result :
        print(out,end='\n')
    print('\n\n---------------------------------标号简记---------------------------------------\n\n')
    listResult=[[] for i in range(k)]
    count=0
    for i in assignments:
        listResult[i].append(count)
        count=count+1
    write_results(listResult,dataset,k)
    for kind in range(k):
        print("第%d类数据有:"%(kind+1))
        count=0
        for j in listResult[kind]:
             print(j,end=' ')
             count=count+1
             if count%25==0:
                 print('\n')
        print('\n')
    print('\n\n--------------------------------------------------------------------------------\n\n')

def main():
    dataset=read_points()
    k_means(dataset,3)

if __name__ == "__main__":   
    main()
```

- **分类结果** 
  a. 通过多次运行程序发现，所得结果与初始值的选定有着密切的关系，并且由于在我的程序中采用随机数的方式产生初值，因此经过观察发现有多种结果。 
  b. 其中两种常见的结果之一如下： 
  第1类数据有:（50） 
  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
  第2类数据有:（38） 
  52 77 100 102 103 104 105 107 108 109 110 111 112 115 116 117 118 120 122 124 125 128 129 130 131 132 134 135 136 137 139 140 141 143 144 145 147 148 
  第3类数据有:（62） 
  50 51 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 
  76 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 101 106 
  113 114 119 121 123 126 127 133 138 142 146 149 
  c. 结果之二： 
  第1类数据有:（50） 
  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 
  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
  第2类数据有:（61） 
  51 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 
  78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 101 106 113 
  114 119 121 123 126 127 133 138 142 146 149 
  第3类数据有:（39） 
  50 52 77 100 102 103 104 105 107 108 109 110 111 112 115 116 117 118 120 122 124 125 128 129 130 131 132 134 135 136 137 139 140 141 143 144 145 147 148