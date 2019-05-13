import automate
import model_utils
from dataset import Dataset, Dataset2
import logging
import pickle

logging.basicConfig(level=logging.INFO)


def main():
    datasets = ['colon', 'leukemia']
    for dataset_name in datasets:
        feature_selection_type = 'select_k_best'
        n_features = 100
        data = Dataset("../Data", dataset_name)

        logging.info("Selecting best features")

        if feature_selection_type is 'select_k_best':
            best_features = model_utils.get_kbest_features(data.X_train,
                                                           data.Y_train,
                                                           n_features)
        else:
            top_features = model_utils.get_kbest_features(data.X_train,
                                                          data.Y_train,
                                                          1000)
            model = model_utils.build_logistic_regression_model(
                data.X_train, data.Y_train, feature_indices=best_features)
            best_features = model_utils.perform_RFE(model, n_features, 10,
                                                    data.X_train, data.Y_train,
                                                    top_features)

        logging.info("Building KNN model on %s", dataset_name)
        # KNN
        model = model_utils.build_nearest_neighbor_model(
            data.X_train, data.Y_train, feature_indices=best_features)
        model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                         data.Y_test, model,
                                         feature_indices=best_features)

        # Naive Bayes
        logging.info("Building Naive Bayes model on %s", dataset_name)
        model = model_utils.build_naive_bayes_model(
            data.X_train, data.Y_train, feature_indices=best_features)
        model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                         data.Y_test, model,
                                         feature_indices=best_features)

        # Logistic Regression
        logging.info("Building Logistic Regression model on %s", dataset_name)
        model = model_utils.build_logistic_regression_model(
            data.X_train, data.Y_train, feature_indices=best_features)
        model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                         data.Y_test, model,
                                         feature_indices=best_features)

        # SVM
        logging.info("Building SVM model on %s", dataset_name)
        model = model_utils.build_svm_model(data.X_train, data.Y_train,
                                            feature_indices=best_features)
        model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                         data.Y_test, model,
                                         feature_indices=best_features)
        model_utils.plot_roc_curve(data.X_train, data.Y_train, data.X_test,
                                   data.Y_test, model,
                                   feature_indices=best_features)

        knn_scores = automate.test_for_all_knn(data)
        gnb_scores = automate.test_for_all_gnb(data)
        rf_scores = automate.test_for_all_rf(data)
        svm_scores = automate.test_for_all_svm(data)
        lr_scores = automate.test_for_all_lr(data)

        scores = dict()
        scores['knn'] = knn_scores
        scores['gnb'] = gnb_scores
        scores['rf'] = rf_scores
        scores['svm'] = svm_scores
        scores['lr_scores'] = lr_scores

        with open("scores_"+dataset_name+".pkl", "wb") as f:
            pickle.dump(scores, f)

    data_dict = dict()
    dataset_wrk = ['breast', 'prostate']
    for d in dataset_wrk:
        data_dict[d] = Dataset2("../Data", d, kfold=10)
    m_list = ['knn', 'lr', 'svm', 'knn', 'rf', 'nb']
    train_scores = dict()
    test_scores = dict()

    for d in data_dict:
        train_scores[d] = dict()
        test_scores[d] = dict()
    
    fsel="kbest"
    for d in data_dict:
        print(d, end=":\n")
        data = data_dict[d]
        for m in ['knn', 'nb', 'rf', 'svm', 'lr']:
            print("\t"+str(m))
            train_scores[d][m], test_scores[d][m] = model_utils.automod(data, mode=m, f_sel=fsel)

    f = open("dump_trainscores_kbest.pkl", "wb")
    pickle.dump(train_scores, f)

    f = open("dump_testscores_kbest.pkl", "wb")
    pickle.dump(test_scores, f)

    f.close()


if __name__ == "__main__":
    main()
