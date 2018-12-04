from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from numpy import mean
import sys
import time
import pickle
import json
import math

def numLines(filename):
    ret = 0
    with open(filename, 'r') as f:
        ret = sum(1 for line in f)
    return ret

def loadData(filename, startLine, endLine):
    data, labels = [], []
    i = 0
    with open(filename, 'r') as f:
        for line in f:
            if i > endLine:
                break
            elif i >= startLine:
                obj = json.loads(line)
                data.append(obj['text'])
                labels.append(obj['stars'])
            i += 1
    return data, labels

class Metric1:
    def __init__(self, metadata_p = 0):
        self.metadata = metadata_p
    def calc(self, original_label, predicted_label_proba):
        return predicted_label_proba[original_label]

class Metric2:
    def __init__(self, metadata_p = 1):
        self.metadata = metadata_p
    def calc(self, original_label, predicted_label_proba):
        wrong_metric = 0.0
        for i in range(1, 6):
            wrong_metric = wrong_metric + ((math.fabs(original_label - i) ** self.metadata) * predicted_label_proba[i])
        return -wrong_metric

def test_metric(metric, original_labels, predicted_label_probas, changed_amount, real_or_not):
    confidence = [(metric.calc(original_labels[i], predicted_label_probas[i]), i) for i in range(len(original_labels))]
    confidence.sort(key = lambda x : x[0])
    hit = 0
    for x in confidence[:changed_amount]:
        if not real_or_not[x[1]]:
            hit += 1
    print("the number of detected wrong reviews in the first", changed_amount, "lowest confidence reviews: ", hit)

def test():
    #posneg = False
    #percentData = 100
    data_file_name = 'yelp_academic_dataset_review.json'
    model_file_name = 'lr_False_100_clf.pickle'
    num_lines = numLines(data_file_name)
    linesToRead = int(num_lines * (float(100) / 100.0))
    train_end = linesToRead * 0.8

    #train_data, train_labels = loadData(filename, 0, train_end, posneg)
    test_data, test_labels = loadData(data_file_name, train_end + 1, linesToRead)

    length = len(test_labels)
    print("the number of test reviews:", length)

    real_or_not = [True for i in range(length)]

    max_changed_amount = 10000
    changed_amount = 0
    for i in range(length):
        if changed_amount >= max_changed_amount:
            break
        if test_labels[i] == 5:
            test_labels[i] = 1
            real_or_not[i] = False
            changed_amount += 1
        elif test_labels[i] == 1:
            test_labels[i] = 5
            real_or_not[i] = False
            changed_amount += 1
    print("the number of manually changed reviews:", changed_amount)

    with open(model_file_name, "rb") as f:
        text_clf = pickle.load(f)

    predicted_label_probas = text_clf.predict_proba(test_data)

    values = [i / 1000.0 for i in range(1, 11)]
    for i in values:
        print(i, end = ':')
        test_metric(Metric2(i), test_labels, predicted_label_probas, changed_amount, real_or_not)

if __name__ == '__main__':
    test()
