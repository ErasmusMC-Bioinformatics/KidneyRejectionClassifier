#!/usr/bin/python3

from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve, precision_recall_curve, \
    average_precision_score
from joblib import dump, load
from numpy import interp

BHOT_FILE = "Data/BHOT_entrez_mapping.csv"
BHOT_PLUS_FILE = "Data/BHOT_plus_entrez_mapping.csv"

CV_OUTER_FOLDS = 10
CV_INNER_FOLDS = 3

# define parameter tuning grid
c_grid_rf = [
     {'n_estimators': [200, 500], 'max_features': [100, 50, 20],  'min_samples_split': [2, 4, 8]}
]

# define numbers for all labels
labels = {"ABMR": 0, "TCMR": 1, "NR": 2}


def timer(start_time=None):
    """Function that determines time used to execute the script"""
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print("Time taken for training model: %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec, 2)))


def load_process_data(expression_file, pheno_file):
    """Function that transforms gene expression and phenotype files in required data matrix formats"""
    # read files into dataframes
    expression_df = pd.read_csv(expression_file, index_col=0)
    # remove index column if present
    try:
        expression_df = expression_df.drop("---", axis=1)
        expression_df = expression_df.drop("Unnamed: 0", axis=1)
    except:
        pass
    # sort dataframe so we know for certain x and y are matched
    expression_df = expression_df.reindex(sorted(expression_df.columns), axis=1)
    sample_ids = expression_df.index.tolist()
    # transform phenotype df into numerical array
    pheno_df = pd.read_csv(pheno_file, index_col=0)
    # sort dataframe to ensure x and y are matched
    pheno_df = pheno_df.reindex(sorted(pheno_df.columns), axis=1)
    pheno_label_num = pheno_df['Label'].map(labels)
    print("Data loaded")
    return expression_df, pheno_label_num, sample_ids


def feature_selection(x_train, y_train, x_test, method):
    """Function that can call different feature selection methods"""
    if method == "bhot" or method == "bhot+":
        x_train = bhot_feature_selection(x_train, method)
        x_test = bhot_feature_selection(x_test, method)
        selected_features = x_train.columns
    elif method == "recursive":
        features = x_train.columns
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=100)
        x_train = sfs.fit_transform(x_train, y_train)
        x_test = sfs.transform(x_test)
        mask = sfs.get_support()
        selected_features = features[mask]
    return x_train, x_test, selected_features


def bhot_feature_selection(x, method):
    """Function that filters a gene expression matrix on genes present in the B-HOT panel"""
    if method == "bhot":
        bhot_df = pd.read_csv(BHOT_FILE, sep=";", dtype=str)
    elif method == "bhot+":
        bhot_df = pd.read_csv(BHOT_PLUS_FILE, sep=";", dtype=str)
    new_x = x[x.columns.intersection(bhot_df["Entrez Gene"])]
    return new_x


def determine_multilabel_aucs(class_labels, y_test, y_pred):
    """Function that determines the AUC scores for each class in multiclass classification"""
    y_test = label_binarize(y_test, classes=range(len(class_labels.keys())))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    classes = class_labels.keys()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc


def determine_multilabel_rocs(class_labels, y_test, y_pred, outer_folds, cv_results_df):
    """Function that generates interpolated ROC curves for cross-validation procedure"""
    index = 0
    colors = get_cmap('tab10').colors
    for label in class_labels:
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        aucs = cv_results_df['AUC: ' + label][:-1]
        for i in range(outer_folds):
            true_labels = y_test[i][:, index]
            pred_probs = y_pred[i][:, index]
            fpr, tpr, thresholds = roc_curve(y_true=true_labels, y_score=pred_probs, pos_label=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid()
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        mean_tprs = np.mean(tprs, axis=0)
        std_tprs = np.std(tprs, axis=0)
        mean_tprs[-1] = 1.0

        mean_auc = np.mean(np.array(aucs))
        std_auc = np.std(aucs)

        tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        tprs_lower = mean_tprs - std_tprs
        plt.plot(mean_fpr, mean_tprs, color=colors[index],
                 label='%s (AUC = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc))
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[index], alpha=0.3)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')
        plt.grid()
        index += 1

    plt.savefig(fname='cv_roc.tiff', dpi=600)


def determine_multilabel_precision(class_labels, y_test, y_pred, outer_folds):
    """Function that generates interpolated Precision Recall curves for cross - validation procedure"""
    fig, ax = plt.subplots()
    index = 0
    colors = get_cmap('tab10').colors
    for label in class_labels:
        y_real = []
        y_proba = []

        precision_array = []
        recall_array = np.linspace(0, 1, 100)
        average_precisions = []

        for i in range(outer_folds):
            true_labels = y_test[i][:, index]
            pred_probs = y_pred[i][:, index]
            precision, recall, thresh = precision_recall_curve(true_labels, pred_probs)
            precision = precision[::-1]
            recall = recall[::-1]
            precision_array = interp(recall_array, recall, precision)
            average_precisions.append(average_precision_score(true_labels, pred_probs))
            y_real.append(true_labels)
            y_proba.append(pred_probs)

        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)

        precision, recall, _ = precision_recall_curve(y_real, y_proba)

        mean_ap = np.mean(average_precisions)
        std_ap = np.std(average_precisions)

        plt.plot(recall, precision, color=colors[index],
                 label='%s (AP = %0.2f $\pm$ %0.2f)' % (label, mean_ap, std_ap))

        std_precision = np.std(precision_array)
        plt.fill_between(recall, precision + std_precision, precision - std_precision, color=colors[index], alpha=0.3)
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")
        plt.grid()

        index += 1

    plt.savefig(fname='cv_precision_recalls.tiff', dpi=600)


def create_performance_report(accuracies, multi_aucs, classification_reports, class_labels, outer_folds, outer_y_test,
                              outer_y_pred):
    """Function that generates dataframe with cross-validation performances"""
    cv_results_df = pd.DataFrame({'CV': range(1, outer_folds + 1), 'Accuracy': accuracies})
    index = 0
    for label in class_labels:
        precision = [classification_reports[i][label]['precision'] for i in range(outer_folds)]
        recall = [classification_reports[i][label]['recall'] for i in range(outer_folds)]
        f1 = [classification_reports[i][label]['f1-score'] for i in range(outer_folds)]
        auc = [multi_aucs[i][index] for i in range(outer_folds)]
        cv_results_df['precision: ' + label], cv_results_df['recall: ' + label], cv_results_df['f1-score: ' + label], \
        cv_results_df['AUC: ' + label] = precision, recall, f1, auc
        index += 1
    # calculate mean of all metrics
    cv_results_df.loc[-1] = cv_results_df.mean()
    cv_results_df.loc[-1, 'CV'] = "Average of %s fold CV" % (outer_folds)
    determine_multilabel_rocs(class_labels, outer_y_test, outer_y_pred, outer_folds, cv_results_df)
    determine_multilabel_precision(class_labels, outer_y_test, outer_y_pred, outer_folds)
    return cv_results_df


def train_model(x, y, c_grid, outer_folds, inner_folds, class_labels, selection_method):
    """Function that trains a multiclass randomforest classifier using a nest cross validation procedure and feature
    selection"""
    # configure the cross-validation procedure
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=1)
    # create lists to store model and cross-validation results
    outer_models, outer_accuracy, outer_params, outer_aucs, outer_specificity, outer_reports, outer_y_test, outer_y_pred = (
        [] for i in range(8))
    current_fold = 1
    for train_ix, test_ix in outer_cv.split(x, y):
        print("Training model on CV fold %s" % (current_fold))
        X_train, X_test = x.iloc[train_ix, :], x.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # execute feature selection method
        X_train, X_test, new_features = feature_selection(X_train, y_train, X_test, method=selection_method)
        print("Features selected")
        # configure the cross-validation procedure
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=1)
        # define the model
        model = RandomForestClassifier(random_state=1)
        # define search space
        grid_search = GridSearchCV(estimator=model, param_grid=c_grid, cv=inner_cv)
        # execute search
        result = grid_search.fit(X_train, y_train)
        # get best performing model fit on the whole training set and store names of features and parameters
        best_model = result.best_estimator_
        best_model.custom_feature_names = new_features
        best_params = result.best_params_
        # evaluate model on the hold out dataset
        y_pred = best_model.predict(X_test)
        y_pred_prod = best_model.predict_proba(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, y_pred)
        y_test = label_binarize(y_test, classes=range(len(class_labels.keys())))
        y_pred = label_binarize(y_pred, classes=range(len(class_labels.keys())))
        outer_y_test.append(y_test)
        outer_y_pred.append(y_pred_prod)
        auc = determine_multilabel_aucs(class_labels, y_test, y_pred_prod)
        # store the results
        outer_accuracy.append(acc)
        outer_aucs.append(auc)
        outer_params.append(best_params)
        outer_models.append(best_model)
        outer_reports.append(classification_report(y_test, y_pred, target_names=class_labels, output_dict=True))
        current_fold += 1
    # determine params of highest scoring model
    best_index = outer_accuracy.index(min(outer_accuracy))
    final_model = outer_models[best_index].fit(x[x.columns.intersection(outer_models[best_index].custom_feature_names)],
                                               y)
    cv_results_df = create_performance_report(outer_accuracy, outer_aucs, outer_reports, class_labels, outer_folds,
                                              outer_y_test, outer_y_pred)

    print("Final model trained with average accuracy of %s" % (cv_results_df['Accuracy'].values[-1]))

    return final_model, cv_results_df


def test_model(model, x_test, y_test, class_labels, cv_results_df):
    """Function that executes testing of best performing model on independent validation set"""
    # execute feature selection first
    x_test = x_test[x_test.columns.intersection(model.custom_feature_names)]

    final_prediction_label = model.predict(x_test)

    final_prediction_one_hot = np.zeros((final_prediction_label.size, final_prediction_label.max() + 1))
    final_prediction_one_hot[np.arange(final_prediction_label.size), final_prediction_label] = 1
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)

    evaluation_table = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    cv_results_df = cv_results_df.append(pd.Series(name='Test', dtype=float))
    cv_results_df.loc['Test', 'CV'] = 'Test'
    cv_results_df.loc['Test', 'Accuracy'] = accuracy_score(y_test, y_pred)

    for label in class_labels:
        precision = evaluation_table[label]['precision']
        recall = evaluation_table[label]['recall']
        f1 = evaluation_table[label]['f1-score']
        cv_results_df.loc['Test', ('precision: ' + label)], cv_results_df.loc['Test', ('recall: ' + label)], \
        cv_results_df.loc['Test', ('f1-score: ' + label)] = precision, recall, f1

    y_test = label_binarize(y_test, classes=[0, 1, 2])

    # Compute ROC curve and Precision Recall curves for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    average_precision = dict()
    classes = class_labels
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        cv_results_df.loc['Test', ('AUC: ' + classes[i])] = roc_auc[i]
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    colors = get_cmap('tab10').colors
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=1,
                 label='ROC curve of class {0} (AUC = {1:0.2f})' ''.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    plt.savefig("test_roc.tiff", dpi=600)
    plt.figure()
    for i, color in zip(range(3), colors):
        plt.plot(recall[i], precision[i], color=colors[i], lw=1,
                 label='Precision-recall of class {0} (AP = {1:0.2f})' ''.format(classes[i], average_precision[i]))
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
    plt.savefig("test_precision.tiff", dpi=600)

    print("Final model test set accuracy of %s" % (cv_results_df['Accuracy'].values[-1]))

    return cv_results_df


def main(args):
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    # process training files
    expression_df, classification_num, sample_ids = load_process_data(args.trainfile,
                                                                      args.trainpheno)

    # Generate model predicting different types of rejection
    rejection_model, cv_results_df = train_model(expression_df, classification_num, c_grid_rf, CV_OUTER_FOLDS,
                                                 CV_INNER_FOLDS, labels, args.selection_method)

    timer(start_time)  # timing ends here for "start_time" variable
    if args.modelfile:
        dump(rejection_model, args.modelfile)

    # Validate models if test files are given
    if args.testfile:
        test_expression_df, test_classification_num, test_sample_ids = load_process_data(
            args.testfile, args.testpheno)
        cv_results_df = test_model(rejection_model, test_expression_df, test_classification_num, ["NR", "TCMR", "ABMR"],
                                   cv_results_df)

    if args.outfile:
        cv_results_df.to_csv(args.outfile, sep="\t", index=False)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_file', help='The gene expression file of the training set', required=True,
                            dest='trainfile')
    arg_parser.add_argument('--train_pheno', help='The phenotype classification file of the training set',
                            required=True, dest='trainpheno')
    arg_parser.add_argument('--test_file', help='The gene expression file of the test set', required=False,
                            dest='testfile')
    arg_parser.add_argument('--test_pheno', help='The phenotype classification file of the test set', required=False,
                            dest='testpheno')
    arg_parser.add_argument('--model', '-m', help='The classification performance report file', required=False,
                            dest='modelfile')
    arg_parser.add_argument('--out', '-o', help='The classification performance report file', required=False,
                            dest='outfile')
    arg_parser.add_argument('--features', '-f',
                            help='The feature selection method. One can chooce "bhot", "recursive" or "bhot+"',
                            required=False, choices=["bhot", "recursive", "bhot+"], dest='selection_method')
    args = arg_parser.parse_args()

    main(args)
