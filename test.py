import numpy as np
import sklearn
import csv
import pandas as pd
import time
import os
import pickle
import math
from sklearn.ensemble import RandomForestClassifier
import re


def entropy(s):
    c, count = np.unique(list(s), return_counts=True)
    n = sum(count)
    en = 0
    for i in count:
        j = i / n
        en -= j * math.log(j, 2)
    return en


def number_count(s):
    f1 = re.findall('(\d)', s)
    return len(f1)


def segmentation(s):
    f1 = s.split('.')
    return len(f1)



if __name__ == '__main__':
    train_file = open("train.txt", "r", encoding='utf-8')
    train_reader = csv.reader(train_file)
    test_file = open("test.txt", "r", encoding='utf-8')
    test_reader = csv.reader(test_file)
    result_file = open("result.txt", "w", encoding='utf-8', newline='')
    result_writer = csv.writer(result_file)

    train_x = []
    train_y = []
    test = []
    test_x = []

    for item in train_reader:
        train_x.append([len(item[0]), number_count(item[0]), entropy(item[0]), segmentation(item[0])])
        if item[1] == 'notdga':
            train_y.append(0)
        else:
            train_y.append(1)

    for item in test_reader:
        test_x.append([len(item[0]), number_count(item[0]), entropy(item[0]), segmentation(item[0])])
        test.append(item[0])

    forest = RandomForestClassifier(n_estimators=2000, random_state=0, n_jobs=-1)
    forest.fit(train_x, train_y)
    res = forest.predict(test_x)

    for i in range(len(res)):
        if res[i] == 0:
            result_writer.writerow([test[i], 'notdga'])
        else:
            result_writer.writerow([test[i], 'dga'])

    train_file.close()
    test_file.close()
    result_file.close()
