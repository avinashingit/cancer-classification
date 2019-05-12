import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import matplotlib.pyplot as plt
from dataset import Dataset
from model_utils import *

logging.basicConfig(level=logging.INFO)


def test_for_all_knn(data):
    from collections import defaultdict
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for nf in range(5, 51):
        for k in range(1, 21):
            best_features = get_kbest_features(data.X_train, data.Y_train, nf)
            model = build_nearest_neighbor_model(data.X_train, data.Y_train, k, best_features)
            train_acc, test_acc = calculate_accuracies(
                data.X_train, data.Y_train, data.X_test, data.Y_test, model, feature_indices=best_features)
            train_scores[nf].append(train_acc)
            test_scores[nf].append(test_acc)
    return train_scores, test_scores


def test_for_all_gnb(data):
    from collections import defaultdict
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for nf in range(40, 250, 20):
        best_features = get_kbest_features(data.X_train, data.Y_train, nf)
        model = build_naive_bayes_model(data.X_train, data.Y_train, feature_indices=best_features)
        train_acc, test_acc = calculate_accuracies(
            data.X_train, data.Y_train, data.X_test, data.Y_test, model, feature_indices=best_features)
        train_scores[1].append(train_acc)
        test_scores[1].append(test_acc)
    return train_scores, test_scores


def test_for_all_rf(data):
    from collections import defaultdict
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for nf in range(5, 51):
        best_features = get_kbest_features(data.X_train, data.Y_train, nf)
        model = build_random_forest_model(data.X_train, data.Y_train, feature_indices=best_features)
        train_acc, test_acc = calculate_accuracies(
            data.X_train, data.Y_train, data.X_test, data.Y_test, model, feature_indices=best_features)
        train_scores[1].append(train_acc)
        test_scores[1].append(test_acc)
    return train_scores, test_scores


def test_for_all_lr(data):
    from collections import defaultdict
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for nf in range(5, 51):
        best_features = get_kbest_features(data.X_train, data.Y_train, nf)
        model = build_logistic_regression_model(
            data.X_train, data.Y_train, feature_indices=best_features)
        train_acc, test_acc = calculate_accuracies(
            data.X_train, data.Y_train, data.X_test, data.Y_test, model, feature_indices=best_features)
        train_scores[1].append(train_acc)
        test_scores[1].append(test_acc)
    return train_scores, test_scores


def test_for_all_svm(data):
    from collections import defaultdict
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    for nf in range(5, 51):
        best_features = get_kbest_features(data.X_train, data.Y_train, nf)
        model = build_svm_model(data.X_train, data.Y_train, feature_indices=best_features)
        train_acc, test_acc = calculate_accuracies(
            data.X_train, data.Y_train, data.X_test, data.Y_test, model, feature_indices=best_features)
        train_scores[1].append(train_acc)
        test_scores[1].append(test_acc)
    return train_scores, test_scores
