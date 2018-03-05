import math
import operator


# 计算数据的信息熵
def calcEntropy(data):
    # 计算数据集中实例的总数
    numSamples = len(data)
    # 创建数据字段
    numClass = {}
    Entropy = 0.0
    # 取键值最后的一列的数值的最后一个字符串
    label = [sample[-1] for sample in data]
    for i in label:
        numClass[i] = numClass.get(i, 0) + 1  # 求不同类的数量
    for j in numClass:
        prob = float(numClass[j] / numSamples)
        # 以低为2求对数
        Entropy = Entropy - prob * math.log(prob, 2)
    return Entropy


# data:待划分的数据集
# i:划分数据集的特征
# setvalue:需要划分的目标值
# 取出数据中第i列值为setValue的样本
def splitData(data, i, setValue):
    subData = []
    for sample in data:
        if sample[i] == setValue:
            reducedSample = sample[:i]  # 删除该样本的第i列
            reducedSample.extend(sample[i + 1:])
            subData.append(reducedSample)
    # 返回第i行值为setvalue的行，并且删除第i行
    return subData


# 选择最优属性对应列的索引
def slctAttribute(data):
    # 计算数据集的经验熵(信息熵)
    allEntropy = calcEntropy(data)
    # 数据集样本的个数
    numSamples = len(data)
    # 数据集属性的个数
    numAttributes = len(data[0]) - 1
    initMI = 0.0
    # 计算互信息，并选出互信息值最大的属性：遍历属性列
    for i in range(numAttributes):
        valueList = [sample[i] for sample in data]  # 拿出数据的第i列
        value = set(valueList)  # 拿出这一列的所有不等值
        # 对应属性列的条件熵
        numEntropy = 0.0
        for j in value:
            # 取出数据中第i列值为setValue的样本
            subData = splitData(data, i, j)
            proportion = float(len(subData) / numSamples)
            Entropy = calcEntropy(subData)
            numEntropy = numEntropy + Entropy * proportion
        MI = allEntropy - numEntropy  # 计算互信息
        # 比较能使互信息最大的属性
        if MI > initMI:
            initMI = MI
            slcAttribute = i
    return slcAttribute


# 属性已遍历到最后一个，取该属性下样本最多的类为叶节点类别标记
def majorVote(classList):
    classCount = {}
    for i in classList:
        # 第一次进入，分别把classList的不同值赋给classCount的键值
        if i not in classCount.keys():
            # 构建键值对，用于对每个classList的不同元素来计数
            classCount[i] = 0
        else:
            classCount[i] += 1
    # 按每个键的键值降序排列
    sortClassCount = sorted(classCount.items, key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def createTree(data, attributes):
    classList = [i[-1] for i in data]  # 取data的最后一列（标签值）
    # count出classList中第一个元素的数目，如果和元素总数相等，那么说明样本全部属于某一类，此时结束迭代
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(data[0]) == 1:  # 遍历后只剩下一个属性，那么标记类别为样本最多的类
        return majorVote(classList)
    selectAttribute = slctAttribute(data)
    bestAttribute = attributes[selectAttribute]
    myTree = {bestAttribute: {}}  # 生成树，采用字典嵌套的方式记录树
    del (attributes[selectAttribute])  # 删除此时的最优属性
    attributeValue = [sample[selectAttribute] for sample in data]  # 取出data所有样本的第selectAttribute个变量的值
    branch = set(attributeValue)  # 取唯一取值，作为本节点的所有分支
    for value in branch:
        subAttributes = attributes[:]
        myTree[bestAttribute][value] = createTree(splitData(data, selectAttribute, value), subAttributes)  # 迭代生成子树
    return myTree


# 读取数据文档中的训练数据（生成二维列表）
def createTrainData():
    lines_set = open('../data/ID3/Dataset.txt').readlines()
    labelLine = lines_set[2]
    labels = labelLine.strip().split()
    lines_set = lines_set[4:11]
    dataSet = []
    for line in lines_set:
        data = line.split()
        dataSet.append(data)
    return dataSet, labels


# 读取数据文档中的测试数据（生成二维列表）
def createTestData():
    lines_set = open('../data/ID3/Dataset.txt').readlines()
    lines_set = lines_set[15:22]
    dataSet = []
    for line in lines_set:
        data = line.strip().split()
        dataSet.append(data)
    return dataSet


# 实用决策树进行分类
def classify(inputTree, featLabels, testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    # 西瓜数据集，根蒂=1代表蜷缩，0代表硬挺；纹理=1代表模糊，0代表清晰
    print("————————————————西瓜数据集——————————————————")
    data = [[1, 0, 'good'], [1, 0, 'good'], [0, 0, 'bad'], [0, 1, 'bad'], [1, 1, 'bad']]
    attributes = ['根蒂', '纹理']
    Tree = createTree(data, attributes)
    print(Tree)
    bootList = ['根蒂', '纹理']
    testList = [[1, 0, 'good'], [1, 0, 'good'], [0, 0, 'bad'], [0, 1, 'bad'], [1, 1, 'bad']]
    i = 1
    for testData in testList:
        dic = classify(Tree, bootList, testData)
        print(i, dic)
        i = i + 1

    # 天气数据
    print("————————————————天气数据集——————————————————")
    myDat, labels = createTrainData()
    Tree = createTree(myDat, labels)
    print(Tree)
    bootList = ['outlook', 'temperature', 'humidity', 'windy']
    testList = createTestData()
    i = 1
    for testData in testList:
        dic = classify(Tree, bootList, testData)
        print(i, dic)
        i = i + 1
