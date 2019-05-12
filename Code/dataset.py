import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


class Dataset:
    def __init__(self, data_dir, dataset_name):

        self.path = os.path.join(data_dir, dataset_name)
        self.dataset_name = dataset_name

        if dataset_name is 'colon':
            self.gene_expression_values_file = "gene_values.txt"
            self.labels_file = "labels_for_each_tissue.txt"

        elif dataset_name is 'leukemia':
            self.scale_file = 'rescale_factors.txt'
            self.samples_file = "table_ALL_AML_samples.txt"
            self.train_file = "train.tsv"
            self.test_file = "test.tsv"

        self.read_data()
        self.transform_data()

    def read_data(self):

        logging.info("Reading Dataset %s", self.dataset_name)

        if self.dataset_name is 'colon':
            with open(os.path.join(self.path, self.gene_expression_values_file), 'r') as f:
                gene_expression_values = [line.strip() for line in tqdm(f.readlines())]
                expressions = []
                for gene in gene_expression_values:
                    if gene != '':
                        expression_values = np.array(gene.split(" "))
                        expressions.append(expression_values)

            with open(os.path.join(self.path, self.labels_file), 'r') as f:
                labels = [int(line.strip()) for line in tqdm(f.readlines())]
                labels = np.array(labels)
                labels[labels > 0] = 1
                labels[labels <= 0] = 0

            self.features = np.array(expressions, dtype=np.float64).T
            self.target = labels
            self.split_data(split_perc=0.2)

        elif self.dataset_name is 'leukemia':
            with open(os.path.join(self.path, self.scale_file), "r") as f:
                x = f.readlines()
                x = [y.strip().split(" ") for y in x]
                scale_factors = [float(y[1]) for y in x]
                train_scale_factors = np.array(scale_factors[:38])
                test_scale_factors = np.array(scale_factors[38:])

            with open(os.path.join(self.path, self.samples_file), "r") as f:
                x = f.readlines()
                labels = []
                for y in x:
                    yx = y.split("\t")
                    labels.append((1 if yx[2].strip() == 'ALL' else 0))
                self.Y_train = labels[:38]
                self.Y_test = labels[38:]

            train_data = pd.read_csv(os.path.join(self.path, self.train_file), sep="\t")
            train_data = np.array(train_data).T
            self.X_train = train_data*train_scale_factors[:, np.newaxis]

            test_data = pd.read_csv(os.path.join(self.path, self.test_file), sep="\t")
            test_data = np.array(test_data).T
            self.X_test = test_data*test_scale_factors[:, np.newaxis]

            self.features = np.vstack((self.X_train, self.X_test))
            self.target = np.append(self.Y_train, self.Y_test)

        logging.info("Reading data completed. The train dataset size is %s", self.X_train.shape)

    def split_data(self, split_perc=0.2):

        logging.info(
            "Splitting the dataset into train and test sets with a split percentage of %s", split_perc)

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(self.features, self.target, test_size=split_perc,
                             random_state=1405, stratify=self.target)

        logging.info(
            "Splitting is completed. The dimensions of the train dataset are %s", self.X_train.shape)

    def transform_data(self):

        logging.info("Standardizing the data to have to zero mean and one variance")

        standard_scaler = StandardScaler()
        standard_scaler.fit(self.X_train)
        self.X_train = standard_scaler.transform(self.X_train)
        self.X_test = standard_scaler.transform(self.X_test)

        logging.info("Standardizing is completed.")


class Dataset2:
    def __init__(self, par_dir, data_tag, kfold=5):

        wdir = os.path.join(par_dir, data_tag)

        gene = os.path.join(wdir, "gene.csv")
        labels = os.path.join(wdir, "labels.csv")

        print("loading dataset:", end=" ")
        print(data_tag, end=" dataset\n")
        with open(gene, 'r') as f:
            features = []
            for l in f.readlines():
                if len(l) > 1:
                    features.append(np.array(l.split(",")))

        lbl = []
        with open(labels, 'r') as f:
            for l in f.readlines():
                lbl.append(int(l[0])-1)
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        for i in range(kfold):
            ret = train_test_split(features, lbl, test_size=0.2, stratify=lbl)
            self.X_train.append(ret[0])
            self.X_test.append(ret[1])
            self.Y_train.append(ret[2])
            self.Y_test.append(ret[3])
            standard_scaler = StandardScaler()
            standard_scaler.fit(self.X_train[i])
            self.X_train[i] = standard_scaler.transform(self.X_train[i])
            self.X_test[i] = standard_scaler.transform(self.X_test[i])
