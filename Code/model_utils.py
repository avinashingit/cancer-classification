import numpy as np
import logging


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


class ModelUtilities:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def get_important_features(self, method, number_of_features):
        if method is 'select_k_best':
            best_indices = SelectKBest(k=number_of_features, score_func=f_classif).fit(
                self.X_train, self.Y_train).get_support(indices=True)
            return best_indices

    def test_for_all(data):
        from collections import defaultdict
        scores = defaultdict(list)
        for nf in range(5, 51):
            for k in range(1, 21):
                best_features = self.get_important_features(
                    self.X_train, self.Y_train, 'select_k_best', nf)
                tr_acc, test_acc = self.build_nearest_neighbor_model(self, k, best_features)
                scores[nf].append(test_acc)
        return scores


def perform_RFE(model, n_features, step, X_train, Y_train, best_features):
    feature_extractor = RFE(model, n_features, step)
    feature_extractor = feature_extractor.fit(X_train[:, best_features], Y_train)
    top_features = np.where(feature_extractor.ranking_ == 1)
    return top_features[0]


def get_kbest_features(X_train, Y_train, number_of_features):
    best_indices = SelectKBest(k=number_of_features, score_func=f_classif).fit(
        X_train, Y_train).get_support(indices=True)
    return best_indices


def build_nearest_neighbor_model(X_train, Y_train, k=5, feature_indices=None):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, Y_train)

    return model


def build_svm_model(X_train, Y_train, feature_indices=None, kernel='rbf'):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]

    model = svm.SVC(C=1.0, kernel=kernel, gamma=0.01, random_state=1405)
    model.fit(X_train, Y_train)

    return model


def build_naive_bayes_model(X_train, Y_train, feature_indices=None):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]

    model = GaussianNB()
    model.fit(X_train, Y_train)

    return model


def build_random_forest_model(X_train, Y_train, feature_indices=None, n_trees=20):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]

    model = RandomForestClassifier(n_estimators=n_trees)
    model.fit(X_train, Y_train)

    return model


def build_logistic_regression_model(X_train, Y_train, feature_indices=None):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]

    model = LogisticRegression(random_state=1405, solver='lbfgs')
    model.fit(X_train, Y_train)

    return model


def calculate_accuracies(X_train, Y_train, X_test, Y_test, model, feature_indices=None):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]
        X_test = X_test[:, feature_indices]

    train_pred_y = model.predict(X_train)
    test_pred_y = model.predict(X_test)

    train_accuracy = metrics.accuracy_score(y_pred=train_pred_y, y_true=Y_train)
    test_accuracy = metrics.accuracy_score(y_pred=test_pred_y, y_true=Y_test)

    return train_accuracy, test_accuracy


def plot_roc_curve(X_train, Y_train, X_test, Y_test, model, feature_indices=None):
    if feature_indices is not None:
        X_train = X_train[:, feature_indices]
        X_test = X_test[:, feature_indices]

    train_pred_y = model.predict(X_train)
    test_pred_y = model.predict(X_test)
    fpr, tpr, threshold = metrics.roc_curve(Y_test, test_pred_y)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def feat_sel(data, nf, mode=None, model_opt="lr"):
    num_fold = len(data.Y_train)
    ret = []
    if mode is None:
        for i in range(num_fold):
            ret.append(None)
    if mode == "kbest":
        for i in range(num_fold):
            ret.append(SelectKBest(k=nf, score_func=f_classif).fit(
                data.X_train[i], data.Y_train[i]).get_support(indices=True))
    else:
        for i in range(num_fold):
            lr_ext = LogisticRegression(solver='lbfgs')
            rfe = RFE(lr_ext, nf, step=0.1)
            ret.append(rfe.fit(data.X_train[i], data.Y_train[i]).get_support(indices=True))
    return ret


def rfmodel(data, nt=20, fid=0, feat=None):
    model = RandomForestClassifier(n_estimators=nt)
    if feat is None:
        model.fit(data.X_train[fid], data.Y_train[fid])
    else:
        model.fit(data.X_train[fid][:, feat], data.Y_train[fid])
    return model


def lrmodel(data, fid=0, feat=None):
    model = LogisticRegression(solver='lbfgs')
    if feat is None:
        model.fit(data.X_train[fid], data.Y_train[fid])
    else:
        model.fit(data.X_train[fid][:, feat], data.Y_train[fid])
    return model


def knnmodel(data, kn=5, fid=0, feat=None):
    model = KNeighborsClassifier(n_neighbors=kn)
    if feat is None:
        model.fit(data.X_train[fid], data.Y_train[fid])
    else:
        model.fit(data.X_train[fid][:, feat], data.Y_train[fid])
    return model


def nbmodel(data, fid=0, feat=None):
    kf = len(data.X_train)
    model = GaussianNB()
    if feat is None:
        model.fit(data.X_train[fid], data.Y_train[fid])
    else:
        model.fit(data.X_train[fid][:, feat], data.Y_train[fid])
    return model


def svmmodel(data, fid=0, feat=None):
    kf = len(data.X_train)
    model = svm.SVC(C=1.0, kernel='rbf', gamma=0.01)
    if feat is None:
        model.fit(data.X_train[fid], data.Y_train[fid])
    else:
        model.fit(data.X_train[fid][:, feat], data.Y_train[fid])
    return model


def automod(data, mode='lr', f_sel="kbest"):
    train_scores = dict()
    test_scores = dict()
    kf = len(data.X_train)
    for nf in range(5, 301, 20):
        bfeat = feat_sel(data, nf, mode=f_sel)
        train_scores[nf] = 0.0
        test_scores[nf] = 0.0
        for i in range(kf):
            if mode == 'nb':
                model = nbmodel(data, fid=i, feat=bfeat[i])
            if mode == 'lr':
                model = lrmodel(data, fid=i, feat=bfeat[i])
            if mode == 'rf':
                model = rfmodel(data, fid=i, feat=bfeat[i])
            if mode == 'svm':
                model = svmmodel(data, fid=i, feat=bfeat[i])
            if mode == 'knn':
                model = knnmodel(data, fid=i, feat=bfeat[i])
            train_acc, test_acc = calculate_accuracies(
                data.X_train[i], data.Y_train[i], data.X_test[i], data.Y_test[i], model, feature_indices=bfeat[i])
            train_scores[nf] += train_acc
            test_scores[nf] += test_acc
        train_scores[nf] /= kf
        test_scores[nf] /= kf
    return train_scores, test_scores
