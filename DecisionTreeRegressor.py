import pandas as pd
import numpy as np


class node:

    def __init__(self, x, y, a, isLeaf=False, dept=0):
        self.data_x = x
        self.data_y = y
        self.attr = a
        self.isLeaf = isLeaf
        self.dataValue = None
        self.branch = dict({})
        self.optAttr = None
        self.t = None
        self.dept = dept

    def createBranch(self, _node, _a):
        self.branch[_a] = _node


class DecisionTreeRegressor:
    def __init__(self, _max_depth=None, _min_split=2):
        self.max_depth = _max_depth
        self.min_split = _min_split

    def getMSE(self, y):
        ymean = np.mean(y)
        return np.mean((y-ymean)**2)

    def getOptSplit(self, x, y, a):
        xy = np.hstack([x, y.reshape([y.shape[0], 1])])
        split = []
        t = []
        for i in a:
            ta = np.unique(x[:, i])
            splittmp = []
            if np.size(ta) == 1:
                split.append(0)
                t.append(ta)
            else:
                for j in ta[0:-1]:
                    xy_div = [xy[xy[:, i] <= j], xy[xy[:, i] > j]]
                    y_div = [k[:, -1] for k in xy_div]
                    splittmp.append(self.getMSE(
                        y)-np.mean(np.array([self.getMSE(k)*np.size(k) for k in y_div])))
                split.append(max(splittmp))
                t.append(ta[np.argmax(splittmp)])
        split = np.array(split)
        return [a[np.argmax(split)], t[np.argmax(split)]]

    def treeGenerate(self, x, y, a, depth=0):
        curNode = node(x, y, a, dept=depth)
        if depth == self.max_depth or np.size(y) <= self.min_split:
            curNode.isLeaf = True
            curNode.dataValue = np.mean(y)
            return curNode

        if np.unique(curNode.data_y).shape[0] == 1:
            curNode.isLeaf = True
            curNode.dataValue = curNode.data_y[0]
            return curNode

        if np.size(a) == 0 or np.size(np.unique(x)) == 1:
            curNode.isLeaf = True
            curNode.dataValue = np.mean(y)
            return curNode

        optAttr, optAttrType = self.getOptSplit(x, y, a)
        mask = np.array([True for i in a])
        mask[np.argwhere(a == optAttr)] = False
        a_div = a[mask]
        xy = np.hstack([x, y.reshape([y.shape[0], 1])])
        for i in range(2):
            xy_div = xy[xy[:, optAttr] <= optAttrType if i ==
                        0 else xy[:, optAttr] > optAttrType]
            x_div = xy_div[:, 0:-1]
            y_div = xy_div[:, -1]
            if np.size(xy_div) == 0:
                branch = node(x_div, y_div, a_div, dept=depth+1, isLeaf=True)
                branch.dataValue = np.mean(y)
                curNode.createBranch(branch, i)
            else:
                curNode.createBranch(
                    self.treeGenerate(x_div, y_div, a_div, depth=depth+1), i)
                curNode.optAttr = optAttr
                curNode.t = optAttrType
        return curNode

    def fit(self, x, y):
        a = np.array([i for i in range(x.shape[1])])
        self.decisionTree = self.treeGenerate(x, y, a)

    def showTree(self, _root):
        if _root.isLeaf:
            print(_root.dataValue)
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
                return predNode.dataValue
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
        rmse = np.sqrt(np.mean((y_pred-y_test)**2))
        return [y_pred, rmse]


if __name__ == '__main__':

    # 读取波士顿房价预测数据集
    dataSet = pd.read_csv('housing.csv')
    dataSet = dataSet.drop('Id', axis=1)

    # 分割波士顿房价预测数据集
    train_df = dataSet.sample(frac=0.7)
    test_df = dataSet[~dataSet.index.isin(train_df.index)]
    x = dataSet.drop('MEDV', axis=1).values
    y = dataSet['MEDV'].values
    x_train = train_df.drop('MEDV', axis=1).values
    y_train = train_df['MEDV'].values
    x_test = test_df.drop('MEDV', axis=1).values
    y_test = test_df['MEDV'].values

    # 训练模型
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)

    # 评估模型
    y_pred, rmse = model.evaluate(x_test, y_test)
    print('模型在波士顿房价预测数据集的测试集上的均方根误差: ', rmse, sep='')

    # 读取歌曲流行度预测数据集
    dataSet = pd.read_csv('song.csv')
    dataSet = dataSet.drop('Id', axis=1)

    # 分割歌曲流行度预测数据集
    train_df = dataSet.sample(frac=0.7)
    test_df = dataSet[~dataSet.index.isin(train_df.index)]
    x = dataSet.drop('popularity', axis=1).values
    y = dataSet['popularity'].values
    x_train = train_df.drop('popularity', axis=1).values
    y_train = train_df['popularity'].values
    x_test = test_df.drop('popularity', axis=1).values
    y_test = test_df['popularity'].values

    # 训练模型
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)

    # 评估模型
    y_pred, rmse = model.evaluate(x_test, y_test)
    print('模型在歌曲流行度预测数据集的测试集上的均方根误差: ', rmse, sep='')
