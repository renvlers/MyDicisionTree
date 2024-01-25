import pandas as pd
import numpy as np


class node:

    def __init__(self, x, y, a, isLeaf=False):
        self.data_x = x
        self.data_y = y
        self.attr = a
        self.isLeaf = isLeaf
        self.dataClass = None
        self.branch = dict({})
        self.optAttr = None
        self.aType = None
        self.t = None

    def createBranch(self, _node, _a):
        self.branch[_a] = _node


class DecisionTreeClassifier:
    def getEnt(self, y):
        p = np.zeros(np.max(y)+1)
        for i in y:
            p[i] += 1
        p /= y.shape[0]
        ent = 0
        for i in p:
            ent += (0 if i == 0 else i*np.log2(i))
        ent = -ent
        return ent

    def getOptGain(self, x, y, a):
        xy = np.hstack([x, y.reshape([y.shape[0], 1])])
        gain = []
        t = []
        for i in a:
            if x[:, i].dtype == 'int64':
                t = None
                xy_div = [xy[xy[:, i] == val] for val in np.unique(xy[:, i])]
                y_div = [j[:, -1].astype('int64') for j in xy_div]
                gain.append(self.getEnt(
                    y)-sum([j.shape[0]*self.getEnt(j)/y.shape[0] for j in y_div]))
            else:
                ta = np.unique(x[:, i])
                gaintmp = []
                if np.size(ta) == 1:
                    gain.append(0)
                    t.append(ta)
                else:
                    for j in ta[0:-1]:
                        xy_div = [xy[xy[:, i] <= j], xy[xy[:, i] > j]]
                        y_div = [k[:, -1].astype('int64') for k in xy_div]
                        gaintmp.append(self.getEnt(
                            y)-sum([k.shape[0]*self.getEnt(k)/y.shape[0] for k in y_div]))
                    gain.append(max(gaintmp))
                    t.append(ta[np.argmax(gaintmp)])
        gain = np.array(gain)
        return [a[np.argmax(gain)], t[np.argmax(gain)] if t != None else None]

    def treeGenerate(self, x, y, a):
        curNode = node(x, y, a)

        if np.unique(curNode.data_y).shape[0] == 1:
            curNode.isLeaf = True
            curNode.dataClass = curNode.data_y[0]
            return curNode

        if np.size(a) == 0 or np.size(np.unique(x)) == 1:
            curNode.isLeaf = True
            counts = np.bincount(curNode.data_y)
            curNode.dataClass = np.argmax(counts)
            return curNode

        optAttr, optAttrType = self.getOptGain(x, y, a)
        mask = np.array([True for i in a])
        mask[np.argwhere(a == optAttr)] = False
        a_div = a[mask]
        xy = np.hstack([x, y.reshape([y.shape[0], 1])])

        if optAttrType == None:
            for i in np.unique(xy[:, optAttr]):
                xy_div = xy[xy[:, optAttr] == i]
                x_div = xy_div[:, 0:-1]
                y_div = xy_div[:, -1].astype('int64')
                if np.size(xy_div) == 0:
                    branch = node(x_div, y_div, a_div, isLeaf=True)
                    counts = np.bincount(y)
                    branch.dataClass = np.argmax(counts)
                    curNode.createBranch(branch, i)
                else:
                    curNode.createBranch(
                        self.treeGenerate(x_div, y_div, a_div), i)
                    curNode.optAttr = optAttr
                    curNode.aType = curNode.data_x[:,
                                                   curNode.optAttr].dtype.name
        else:
            for i in range(2):
                xy_div = xy[xy[:, optAttr] <= optAttrType if i ==
                            0 else xy[:, optAttr] > optAttrType]
                x_div = xy_div[:, 0:-1]
                y_div = xy_div[:, -1].astype('int64')
                if np.size(xy_div) == 0:
                    branch = node(x_div, y_div, a_div, isLeaf=True)
                    counts = np.bincount(y)
                    branch.dataClass = np.argmax(counts)
                    curNode.createBranch(branch, i)
                else:
                    curNode.createBranch(
                        self.treeGenerate(x_div, y_div, a_div), i)
                    curNode.optAttr = optAttr
                    curNode.aType = curNode.data_x[:,
                                                   curNode.optAttr].dtype.name
                    curNode.t = optAttrType
        return curNode

    def fit(self, x, y):
        a = np.array([i for i in range(x.shape[1])])
        self.decisionTree = self.treeGenerate(x, y, a)

    def showTree(self, _root):
        if _root.isLeaf:
            print(_root.dataClass)
        else:
            print(_root.optAttr, ' = ?', sep='')
            for i in _root.branch.items():
                print(_root.optAttr, ' = ', i[0], ': ', end='', sep='')
                self.showTree(i[1])

    def display(self):
        root = self.decisionTree
        self.showTree(root)

    def predict(self, x):
        predNode = self.decisionTree
        while predNode:
            if predNode.isLeaf:
                return predNode.dataClass
            else:
                if predNode.aType == 'int64':
                    predNode = predNode.branch[x[predNode.optAttr]]
                else:
                    if x[predNode.optAttr] <= predNode.t:
                        predNode = predNode.branch[0]
                    else:
                        predNode = predNode.branch[1]

    def evaluate(self, x, y_test):
        y_pred = []
        for i in x:
            y_pred.append(self.predict(i))
        y_pred = np.array(y_pred)
        acc = sum(np.where(y_pred == y_test, 1, 0))/np.size(y_test)
        return [y_pred, acc]


if __name__ == '__main__':
    model = DecisionTreeClassifier()

    # 读取鸢尾花数据集
    dataSet = pd.read_csv('Iris.csv')
    dataSet = dataSet.drop('Id', axis=1)

    # 分割鸢尾花数据集
    train_df = dataSet.groupby('Species', group_keys=False).apply(
        lambda x: x.sample(frac=0.7))
    test_df = dataSet[~dataSet.index.isin(train_df.index)]
    x = dataSet.drop('Species', axis=1).values
    y = dataSet['Species'].values
    x_train = train_df.drop('Species', axis=1).values
    y_train = train_df['Species'].values
    x_test = test_df.drop('Species', axis=1).values
    y_test = test_df['Species'].values

    # 预处理鸢尾花数据集
    y[y == 'Iris-setosa'] = 0
    y[y == 'Iris-versicolor'] = 1
    y[y == 'Iris-virginica'] = 2
    y = y.astype('int64')
    y_train[y_train == 'Iris-setosa'] = 0
    y_train[y_train == 'Iris-versicolor'] = 1
    y_train[y_train == 'Iris-virginica'] = 2
    y_train = y_train.astype('int64')
    y_test[y_test == 'Iris-setosa'] = 0
    y_test[y_test == 'Iris-versicolor'] = 1
    y_test[y_test == 'Iris-virginica'] = 2
    y_test = y_test.astype('int64')

    # 训练模型
    model.fit(x_train, y_train)

    # 评估模型
    y_pred, acc = model.evaluate(x_test, y_test)
    print('模型在鸢尾花数据集的测试集上的精度: ', acc, sep='')

    # 读取UCI Glass数据集
    dataSet = pd.read_csv('glass.csv')
    dataSet = dataSet.drop('Id', axis=1)

    # 分割UCI Glass数据集
    train_df = dataSet.groupby('Type', group_keys=False).apply(
        lambda x: x.sample(frac=0.8))
    test_df = dataSet[~dataSet.index.isin(train_df.index)]
    x = dataSet.drop('Type', axis=1).values
    y = dataSet['Type'].values
    x_train = train_df.drop('Type', axis=1).values
    y_train = train_df['Type'].values
    x_test = test_df.drop('Type', axis=1).values
    y_test = test_df['Type'].values

    # 训练模型
    model.fit(x_train, y_train)

    # 评估模型
    y_pred, acc = model.evaluate(x_test, y_test)
    print('模型在UCI Glass数据集的测试集上的精度: ', acc, sep='')
