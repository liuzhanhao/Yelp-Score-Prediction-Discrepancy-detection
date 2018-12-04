from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from numpy import mean
from sklearn.model_selection import KFold
import sys
import time
import json
import pickle, csv
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


def loadData():
    data, labels = [], []
    i = 0

    with open('hotel_data.csv', 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        for line in reader:
            review = line[1].strip()
            # for i in range(10):
            #     review = review + ' ' + line[1]
            data.append(review)
            labels.append(line[0])

    return data, labels


def classify(technique, X_train, y_train, X_test, y_test):

    # X, y = loadData()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if technique == 'nb':
        clf_obj = MultinomialNB()
    elif technique == 'svm':
        clf_obj = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)
        #clf_obj = SVC(gamma='auto')
    elif technique == 'lr':
        # 'newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
        clf_obj = LogisticRegression(solver='liblinear')
    elif technique == 'dtree':
        clf_obj = DecisionTreeClassifier()
    elif technique == 'rf':
        clf_obj = RandomForestClassifier()
    elif technique == 'mlp':
        clf_obj  = MLPClassifier(solver='lbfgs', learning_rate='adaptive', alpha=1e-4, hidden_layer_sizes=(10, 10), random_state=42)

    start_time = time.time()
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf_obj),
                         ])

    text_clf = text_clf.fit(X_train, y_train)


    y_predict = text_clf.predict(X_test)
    # print("time: %s seconds" % (time.time() - start_time))

    accuracy = mean(y_predict == y_test)
    print("classification_report: \n ", metrics.classification_report(y_test, y_predict))
    # print("confusion_matrix:\n ", metrics.confusion_matrix(y_test, y_predict))
    return accuracy, metrics.confusion_matrix(y_test, y_predict)


def print_usage():
    print("Usage: classify.py <nb/svm/lr>")
    print("e.g., python classify.py nb")


if __name__ == '__main__':
    techniques = {'nb': 'Naive Bayes', 'svm': 'Support Vector Machines',
                  'lr': 'Logistic Regression', 'dtree': 'Decision Tree', 'rf': 'Random Forest', 'mlp' : 'mlp'}

    try:
        technique = sys.argv[1].lower()  # nb or svm or lr or dtree or rf
    except IndexError:
        print_usage()
        sys.exit(1)

    print("Technique:", techniques[technique])
    X, y = loadData()

    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    confusion_matrix = np.array([[0, 0], [0, 0]])
    accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        accu, matrix = classify(technique, X_train, y_train, X_test, y_test)
        confusion_matrix += matrix
        accuracy += accu

    print("accuracy: %.3f" % (accuracy / 5))
    print("confusion_matrix:\n", confusion_matrix)
    # confusion_matrix: [[TP, FN], [FP, TN]] for Deceptive
    TP_d = confusion_matrix[0][0]
    FN_d = confusion_matrix[0][1]
    FP_d = confusion_matrix[1][0]
    TN_d = confusion_matrix[1][1]
    # confusion_matrix: [[TN, FP], [FN, TP]] for truthful
    TP_t = confusion_matrix[1][1]
    FN_t = confusion_matrix[1][0]
    FP_t = confusion_matrix[0][1]
    TN_t = confusion_matrix[0][0]

    # precision
    deceptive_precision = TP_d / (TP_d + FP_d) # Precision = TP/(TP+FP)
    truthful_precision = TP_t / (TP_t + FP_t)
    print("Deceptive Precision: %.3f" % deceptive_precision)
    print("Truthful Precision: %.3f" % truthful_precision)
    print("Average Precision: %.3f" % ((deceptive_precision + truthful_precision) / 2))
    # recall
    deceptive_recall = TP_d / (TP_d + FN_d) # recall = TP/(TP+FN)
    truthful_recall = TP_t / (TP_t + FN_t)
    print("Deceptive Recall: %.3f" % deceptive_recall)
    print("Truthful Recall: %.3f" % truthful_recall)
    print("Average Recall: %.3f" % ((deceptive_recall + truthful_recall) / 2))
    # f1 score
    deceptive_f1score = 2 * TP_d / (2 * TP_d + FP_d + FN_d) # f1score = 2TP/(2TP+FP+FN)
    truthful_f1score = 2 * TP_t / (2 * TP_t + FP_t + FN_t)
    print("Deceptive F1 score: %.3f" % deceptive_f1score)
    print("Truthful F1 score: %.3f" % truthful_f1score)
    print("Average F1 score: %.3f" % ((deceptive_f1score + truthful_f1score) / 2))
    


