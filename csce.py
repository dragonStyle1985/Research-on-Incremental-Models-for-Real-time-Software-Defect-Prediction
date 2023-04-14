"""
Cost sensitive software defect prediction
"""
from typing import List

from sklearn import tree
import random
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


def ensemble_tree(num):
    """
    Building an ensemble classifier
    :param num:
    :return:
    """
    tree_list_ = []
    for i in range(num):
        tree_i = tree.DecisionTreeClassifier()
        tree_list_.append(tree_i)
    return tree_list_


def ensemble_hybrid_tree(num, sgd_n):
    """
    Building a hybrid ensemble classifier

    :param sgd_n:
    :param num:
    :return:
    """
    tree_list_ = []
    for i in range(num):
        if i < sgd_n:
            tree_i = SGDClassifier()
        else:
            tree_i = tree.DecisionTreeClassifier()
        tree_list_.append(tree_i)
    return tree_list_


def building_under_sampling_data(xdata, ydata, tree_list: List[DecisionTreeClassifier]):
    pos_data_x = []
    pos_data_y = []
    neg_data_x = []
    neg_data_y = []
    print(len(xdata), len(ydata))
    for i in range(len(xdata)):
        if ydata[i] == 1:
            pos_data_x.append(xdata[i])
            pos_data_y.append(ydata[i])
        else:
            neg_data_x.append(xdata[i])
            neg_data_y.append(ydata[i])
    pos_num = len(pos_data_x)
    print(f'pos_num:{len(pos_data_x)}, neg_num:{len(neg_data_x)}')
    if len(neg_data_x) > len(pos_data_x):
        sampled_neg_num = pos_num
    else:
        raise NotImplemented
    for j in range(len(tree_list)):
        random_list = random.sample(range(0, len(neg_data_x)), sampled_neg_num)
        for k in random_list:
            pos_data_x.append(neg_data_x[k])
            pos_data_y.append(neg_data_y[k])
        tree_list[j].fit(pos_data_x, pos_data_y)
        del pos_data_x[pos_num - 1:]
        del pos_data_y[pos_num - 1:]
    return tree_list


def prediction(tree_list_, testdata):
    print_list = []
    predict_list = []
    for i in tree_list_:
        # print(i.predict(testdata))
        predict_list.append(i.predict(testdata))
        # print(i.predict_proba(testdata))
    for j in range(len(testdata)):
        count = 0
        for k in range(len(tree_list_)):
            count = predict_list[k][j] + count
        # print(count)
        if count >= (len(testdata) / 6.0):
            print_list.append(1)
        else:
            print_list.append(0)
    return print_list
