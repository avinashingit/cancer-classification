import automate
import model_utils
from dataset import Dataset
import logging

logging.basicConfig(level=logging.INFO)


def main():
    dataset_name = 'colon'
    feature_selection_type = 'select_k_best'
    n_features = 100
    data = Dataset("../Data", dataset_name)

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
    # KNN
    model = model_utils.build_nearest_neighbor_model(
        data.X_train, data.Y_train, feature_indices=best_features)
    model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                     data.Y_test, model,
                                     feature_indices=best_features)

    # Naive Bayes
    model = model_utils.build_naive_bayes_model(
        data.X_train, data.Y_train, feature_indices=best_features)
    model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                     data.Y_test, model,
                                     feature_indices=best_features)

    # Logistic Regression
    model = model_utils.build_logistic_regression_model(
        data.X_train, data.Y_train, feature_indices=best_features)
    model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                     data.Y_test, model,
                                     feature_indices=best_features)

    # Logistic Regression
    model = model_utils.build_svm_model(data.X_train, data.Y_train,
                                        feature_indices=best_features)
    model_utils.calculate_accuracies(data.X_train, data.Y_train, data.X_test,
                                     data.Y_test, model,
                                     feature_indices=best_features)
    model_utils.plot_roc_curve(data.X_train, data.Y_train, data.X_test,
                               data.Y_test, model,
                               feature_indices=best_features)

    knn_train_scores, knn_test_scores = automate.test_for_all_knn(data)
    gnb_train_scores, gnb_test_scores = automate.test_for_all_gnb(data)
    rf_train_scores, rf_test_scores = automate.test_for_all_rf(data)
    svm_train_scores, svm_test_scores = automate.test_for_all_svm(data)
    lr_train_scores, lr_test_scores = automate.test_for_all_lr(data)
    
    data_dict = dict()
    dataset_wrk = ['breast', 'prostate'] 
    for d in dataset_wrk:
        data_dict[d] = Dataset2("../Data", d, kfold=10) 
    m_list = ['knn','lr','svm','knn','rf','nb']
    train_scores = dict()
    test_scores = dict()

    for d in data_dict:
        train_scores[d] = dict()
        test_scores[d] = dict()

    for d in data_dict:
        print(d,end=":\n")
        data = data_dict[d]
        for m in ['knn', 'nb', 'rf', 'svm', 'lr']:
            print("\t"+str(m))
            train_scores[d][m], test_scores[d][m]  = automod(data, mode=m)
    import pickle

    f = open("dump_trainscores_kbest.pkl","wb")
    pickle.dump(train_scores,f)

    f = open("dump_testscores_kbest.pkl","wb")
    pickle.dump(test_scores,f)

    f.close()
    plt_titledict = {"knn":"KNN", "nb":"Naive Bayes", "rf":"Random Forest", "svm":"SVM", "lr":"Logistic Regression"}
    for d in data_dict:
        print(d,end=":\n")
        data = data_dict[d]

        for m in ['nb', 'rf', 'svm', 'lr', 'knn']:
            plt.figure()
            plt.plot(list(test_scores[d][m].keys()), list(test_scores[d][m].values()), label = "Test")
            plt.title("Accuracy of " +str(plt_titledict[m]))
            plt.xlabel("Number of genes")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(str("Plots/")+str(d)+"_"+str(m)+".jpg"); 
            plt.show();


if __name__ == "__main__":
    main()
