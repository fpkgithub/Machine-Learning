#[【机器学习实战-python3】使用Apriori算法进行关联 分析](http://blog.csdn.net/sinat_17196995/article/details/71124284)

转载：http://blog.csdn.net/sinat_17196995/article/details/71124284

一、背景 
从大规模数据集中寻找物品间的隐含关系被称作**关联分析(association analysis)**或者关联规则学习(association rule learning)。 
关联分析是一种在大规模数据集中寻找有趣关系的任务。这些关系可以有两种形式:频繁项 
集或者关联规则。**频繁项集**(frequent item sets)是经常出现在一块的物品的集合,**关联规则**(association rules)暗示两种物品之间可能存在很强的关系。

当寻找频繁项集时,频繁(frequent)的定义是什么? 
有许多概念可以解答上述问题,不过其中最重要的是**支持度和可信度**。 
1-一个项集的**支持度**(support) 
被定义为数据集中包含该项集的记录所占的比例。支持度是针对项集来说的,因此可以定义一个最小支持度,而只保留满足最 
小支持度的项集。

2-**可信度或置信度**(confidence) 
是针对一条诸如{尿布} ➞ {葡萄酒}的关联规则来定义的。这 
条规则的可信度被定义为“支持度({尿布, 葡萄酒})/支持度({尿布})”。从图11-1中可以看到,由 
于{尿布, 葡萄酒}的支持度为3/5,尿布的支持度为4/5,所以“尿布 ➞ 葡萄酒”的可信度为3/4=0.75。 
这意味着对于包含“尿布”的所有记录,我们的规则对其中75%的记录都适用。

如果面对成千上万的数据，如生成一个物品所有可能组合的清单,然后对每一种组合统计它出现的频繁程度,但当物品成千上万时,上述做法非常非常慢。**这里就需要引入Apriori原理来减少计算量。**

二、Apriori 原理与实现 
Apriori原理可以帮我们减少可能感兴趣的项集。Apriori原理是说如果某个项集是频繁的,那么它的所有子集也是频繁的。这个原理直观上并没有什么帮助,但是如果反过来看就有用了,也就是说如果一个项集是非频繁集,那么它的所有超集也是非频繁的。 
![这里写图片描述](http://img.blog.csdn.net/20170503152655023?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

关联分析的目标包括两项:发现频繁项集和发现关联规则。首先需要找到频繁 
项集,然后才能获得关联规则。 
**Apriori算法是发现频繁项集的一种方法。** Apriori算法的两个输入参数分别是最小支持度和数 
据集。该算法首先会生成所有**单个**物品的项集列表。接着扫描交易记录来查看哪些项集满足最小支持度要求,那些**不满足最小支持度的集合会被去掉**。然后,对剩下来的集合进行组合以生成包含两个元素的项集。接下来,再重新扫描交易记录,去掉不满足最小支持度的项集。该过程重复进行直到所有项集都被去掉。

```python
# -*- coding: utf-8 -*-

from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) #store all the item unrepeatly

    C1.sort()
    #return map(frozenset, C1)#frozen set, user can't change it.
    return list(map(frozenset, C1))
```

函数 loadDataSet() 创建了一个用于测试的简单数据集； 
函数 createC1() 将构建集合 C1 。 
*Apriori算法首先构建集合 C1 ,然后扫描数据集判断这些只有一个元素的项集是否满足最小支持度的要求。那些满足最低要求的项集构成集合 L1 。而 L1 中的元素相互组合构成 C2 , C2 再进一步过滤变为 L2 。到这里,我想读者应该明白了该算法的主要思路。* 
因此算法需要一个函数 createC1() 来构建第一个候选项集的列表 C1 。由于算法一开始是从输入数据中提取候选项集列表,所以这里需要一个特殊的函数来处理,而后续的项集列表则是按一定的格式存放的。这里使用的格式就是Python中frozenset类型。frozenset是指被“冰冻”的集合,即用户不能修改它们。

```python
#该函数用于从 C1 生成 L1 。
def scanD(D,Ck,minSupport):
#参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
    ssCnt={}
    for tid in D:#遍历数据集
        for can in Ck:#遍历候选项
            if can.issubset(tid):#判断候选项中是否含数据集的各项
                #if not ssCnt.has_key(can): # python3 can not support
                if not can in ssCnt:
                    ssCnt[can]=1 #不含设为1
                else: ssCnt[can]+=1#有则计数加1
    numItems=float(len(D))#数据集大小
    retList = []#L1初始化
    supportData = {}#记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems#计算支持度
        if support >= minSupport:
            retList.insert(0,key)#满足条件加入L1中
        supportData[key] = support
    return retList, supportData
```

测试： 
![这里写图片描述](http://img.blog.csdn.net/20170504084600104?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

上述4个项集构成了 L1 列表,该列表中的每个单物品项集至少出现在50%以上的记录中。由于物品4并没有达到最小支持度,所以没有包含在 L1 中。通过去掉这件物品,减少了查找两物品项集的工作量。

完整的Aprior实现

整个Apriori算法的伪代码如下:

```python
当集合中项的个数大于0时:
    构建一个k个项组成的候选项集的列表
    检查数据以确认每个项集都是频繁的
    保留频繁项集并构建k+1项组成的候选项集的列表(向上合并)1234
```

```python
#total apriori
def aprioriGen(Lk, k): #组合，向上合并
    #creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #两两组合遍历
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

#apriori
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet)) #python3
    L1, supportData = scanD(D, C1, minSupport)#单项最小支持度判断 0.5，生成L1
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):#创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = aprioriGen(L[k-2], k)#Ck
        Lk, supK = scanD(D, Ck, minSupport)#get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
```

主函数是 apriori() ,它会调用 aprioriGen() 来创建候选项集 Ck 。 
**函数 aprioriGen()** 的输入参数为频繁项集列表 Lk 与项集元素个数 k ,输出为 Ck 。举例来说,该函数以{0}、{1}、{2}作为输入,会生成{0,1}、{0,2}以及{1,2}。要完成这一点,首先创建一个空列表,然后计算 Lk 中的元素数目。通过循环来比较 Lk 中的每一个元素与其他元素，紧接着,取列表中的两个集合进行比较。如果这两个集合的前面 k-2 个元素都相等,那么就将这两个集合合成一个大小为 k 的集合 。这里使用集合的并操作来完成。

**apriori函数**首先创建 C1 然后读入数据集将其转化为 D (集合列表)来完 
成。程序中使用 map 函数将 set() 映射到 dataSet 列表中的每一项。scanD() 函数来创建 L1 ,并将 L1 放入列表 L 中。 L 会包含 L1 、 L2 、 L3 …。现在有了 L1 ,后面会继续找 L2 , L3 …,这可以通过 while 循环来完成,它创建包含更大项集的更大列表,直到下一个大的项集为空。Lk 列表被添加到 L ,同时增加 k 的值,增大项集个数，重复上述过程。最后,当 Lk 为空时,程序返回 L 并退出。 
测试： 
![这里写图片描述](http://img.blog.csdn.net/20170504102306469?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
这里的6个集合是候选项集 Ck 中的元素。其中4个集合在 L[1] 中,剩下2个集合被函数 scanD()过滤掉。 
![这里写图片描述](http://img.blog.csdn.net/20170504104244175?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
下面再尝试一下70%的支持度: 
![这里写图片描述](http://img.blog.csdn.net/20170504104407950?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
变量 suppData 是一个字典,它包含我们项集的支持度值。现在暂时不考虑这些值,不过下一节会用到这些值。 
现在可以知道哪些项出现在70%以上的记录中,还可以基于这些信息得到一些结论。我们可以像许多程序一样利用数据得到一些结论,或者可以生成 if-then 形式的关联规则来理解数据。下一节会就此展开讨论。

三、从频繁项集中挖掘关联规则 
人们最常寻找的两个目标是频繁项集与关联规则。上一节介绍如何使用Apriori算法来发现频繁项集,现在需要解决的问题是如何找出关联规则。

对于关联规则,我们也有类似的量化方法,这种量化指标称为可信度。一条规则P ➞ H的可信度定义为 support(P |H)/support(P) 。记住,在Python中,操作符 | 表示集合的并操作。P | H 是指所有出现在集合 P 或者集合 H 中的元素。 
![这里写图片描述](http://img.blog.csdn.net/20170504105102085?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
可以观察到,如果某条规则并不满足最小可信度要求,那么该规则的**所有子集**也不会满足最小可信度要求。以上图为例,假设规则0,1,2 ➞ 3并不满足最小可信度要求,那么就知道任何左部为{0,1,2}子集的规则也不会满足最小可信度要求。 
**可以利用关联规则的上述性质属性来减少需要测试的规则数目。**

可以首先从一个频繁项集开始,接着创建一个规则列表,其中规则右部只包含一个 
元素,然后对这些规则进行测试。接下来合并所有剩余规则来创建一个新的规则列表,其中规则右部包含两个元素。这种方法也被称作**分级法**。 
从上节测试可以看出L频繁项集L[0]为只有单个元素的项集。

```python
#生成关联规则
def generateRules(L, supportData, minConf=0.7):
    #频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = [] #存储所有的关联规则
    for i in range(1, len(L)):  #只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
            #如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)# 调用函数2
    return bigRuleList

#生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    #针对项集中只有两个元素时，计算可信度
    prunedH = []#返回一个满足最小可信度要求的规则列表
    for conseq in H:#后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] #可信度计算，结合支持度数据
        if conf >= minConf:
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            #如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet-conseq, conseq, conf))#添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)#同样需要放入列表到后面检查
    return prunedH

#合并
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m+1)#存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):    
        #满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
```

测试： 
![这里写图片描述](http://img.blog.csdn.net/20170504151733591?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
从生成的规则可以看出2和5之间的前后件可以互换，而1和3不可以。 
修改最小可信度值，再次测试： 
![这里写图片描述](http://img.blog.csdn.net/20170504151927391?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
一旦降低可信度阈值,就可以获得更多的规则。

四、示例:发现毒蘑菇的相似特征 
首先分析一下毒蘑菇的数据集： 
![这里写图片描述](http://img.blog.csdn.net/20170504153125908?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

第一个特征表示有毒或者可食用。如果某样本有毒,则值为2。如果可食用,则值为1。

下一个特征是蘑菇伞的形状,有六种可能的值，分别用整数3-8来表示。 
毒蘑菇中存在的公共特征,可以运行Apriori算法来寻找包含特征值为2的频繁项集。找到特征为毒蘑菇的最关联特征（1个） 
![这里写图片描述](http://img.blog.csdn.net/20170504155351440?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

扩大范围到3个： 
![这里写图片描述](http://img.blog.csdn.net/20170504155547719?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMTcxOTY5OTU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

根据关联规则可以找到与指定特征相关且最频繁的其他特征，从而在各类场景下帮助分析判断。

总结 
关联分析是用于发现大数据集中元素间有趣关系的一个工具集,可以采用两种方式来量化这些有趣的关系。 
第一种方式是使用频繁项集,它会给出经常在一起出现的元素项。 
第二种方式是关联规则,每条关联规则意味着元素项之间的“如果……那么”关系。

**Apriori的方法简化了计算量，在合理的时间范围内找到频繁项集：** 
Apriori原理是说如果一个元素项是不频繁的,那么那些包含该元素的超集也是不频繁的。 
Apriori算法从单元素项集开始,通过组合满足最小支持度要求的项集来形成更大的集合。支持度用来度量一个集合在原始数据中出现的频率。

