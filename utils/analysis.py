import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y, ylim=None, cv=None, scoring="accuracy", n_jobs=-1,
                        train_sizes=np.linspace(.1, 1.0, 10), random_state=None):
    """
    Generate a simple plot of the test and training learning curve.

    Args:
        estimator: object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y: tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
        ylim: tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.
        cv: int, CV generator or an iterable, optional
            Determines the CV splitting strategy.
            Possible inputs for cv are:
                - None, to use the default 3-fold CV,
                - integer, to specify the number of folds.
                - An object to be used as a CV generator.
                - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <model_selection>` for the various
            cross validators that can be used here.
        scoring: callable or None, optional, default: 'accuracy'.
            Refer :ref:`User Guide <sklearn.metrics.*>` for the various
            scoring that can be used here.
        n_jobs:
        train_sizes:
        random_state:

    Returns:
    """

    plt.figure()
    plt.title(str(scoring) + " vs. training set size")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Train Size')
    plt.ylabel(str(scoring))

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="CV Train Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="CV Test Score")

    plt.legend(loc="best")
    plt.show()

    return plt


def print_scores(clf, x_train, y_train, x_test, y_test, n_jobs=-1):
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print("Accuracy Train: %0.5f" % (accuracy_score(y_train, y_train_pred)))
    print("Accuracy Test: %0.5f" % (accuracy_score(y_test, y_test_pred)))

    if len(np.unique(y_train)) == 2:
        print("AUC Train: %0.5f" % (roc_auc_score(y_train, y_train_pred)))
        print("AUC Test: %0.5f" % (roc_auc_score(y_test, y_test_pred)))

    print("f1 Train: %0.5f" % (f1_score(y_train, y_train_pred)))
    print("f1 Test: %0.5f" % (f1_score(y_test, y_test_pred)))

    print("\nConfusion Matrix on the Test data:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report on the Test data:")
    print(classification_report(y_test, y_test_pred))

    return


def print_grid_result(clf, x_train, y_train, param_grid, n_splits=10, scoring=None, random_state=None):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(clf, param_grid, scoring=scoring, n_jobs=-1, cv=kfold)
    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return


def get_dummy_index(string_list, uniqe_list):
    uniqe_inx = pd.Index(uniqe_list)
    string_inx = np.zeros((len(string_list)), dtype=int)

    for i in range(len(string_list)):
        string_inx[i] = uniqe_inx.get_loc(string_list[i])

    return string_inx.tolist()


def get_dummy_features(index_list, cols_out):
    if max(index_list) != (len(cols_out) - 1):
        print("max(index_list) must be the same as len(cols_out)-1.")
        return
    else:
        dummy_features = np.zeros((len(index_list), len(cols_out)), dtype=int)

        for i in range(len(index_list)):
            dummy_features[i][index_list[i]] = 1

        df_features = pd.DataFrame(dummy_features, columns=cols_out)

        return df_features


def check_metric_type(metric_type):
    if metric_type == "accuracy":
        metric = accuracy_score
    elif metric_type == "f1":
        metric = f1_score
    elif metric_type == "recall":
        metric = recall_score
    elif metric_type == "precision":
        metric = precision_score
    else:
        sys.exit("The metric_type must be one of the following in the list: ['accuracy', 'f1', 'recall', 'precision'].")

    return metric


def get_post_processed_arrays(y_true, y_pred, y_prob, confidence_level):
    y_true_post = []
    y_pred_post = []
    for i, prob in enumerate(y_prob):
        if prob >= confidence_level:
            y_true_post.append(y_true[i])
            y_pred_post.append(y_pred[i])

    return y_true_post, y_pred_post


def get_remainig_and_score(y_true, y_pred, y_prob, confidence_level=0.95, metric_type="accuracy"):
    metric = check_metric_type(metric_type)
    y_true_post, y_pred_post = get_post_processed_arrays(y_true, y_pred, y_prob, confidence_level=confidence_level)

    remainig = len(y_true_post) / len(y_true)

    if metric_type == "accuracy":
        score = metric(y_true_post, y_pred_post)
    else:
        score = metric(y_true_post, y_pred_post, average="micro")

    return remainig, score


def plot_score_vs_confidence_level(clf, X, y, confidences, metric_type="accuracy"):
    probabilities = clf.predict_proba(X)
    y_prob = [max(arr_i) for arr_i in probabilities]
    y_pred = clf.predict(X)

    remainigs = []
    scores = []
    for c_i in confidences:
        remainig, score = get_remainig_and_score(y_true=y,
                                                 y_pred=y_pred,
                                                 y_prob=y_prob,
                                                 confidence_level=c_i,
                                                 metric_type=metric_type)
        remainigs.append(remainig)
        scores.append(score)

    plt.figure()
    plt.title("(" + metric_type + " & Remainig % of the Sample) VS. Confidence Level")
    plt.xlabel("Confidence Level")
    plt.ylabel(metric_type + " & Remainig % of the Sample")

    plt.grid(True)

    plt.plot(confidences, remainigs, 'ro-', label="Remainig % of the Sample")
    plt.plot(confidences, scores, 'bo-', label=metric_type)

    plt.legend(loc="best")
    plt.show()

    return plt


def plot_confidence_level_4a_class(df, predicted_col,
                                   label_col, probability_col,
                                   Class, confidences=np.arange(0.00, 0.96, 0.01),
                                   metric_type="accuracy"):
    y_true = df.loc[df[label_col] == Class, label_col].values
    y_pred = df.loc[df[label_col] == Class, predicted_col].values
    y_prob = df.loc[df[label_col] == Class, probability_col].values

    remainigs = []
    scores = []
    for c_i in confidences:
        remainig, score = get_remainig_and_score(y_true=y_true,
                                                 y_pred=y_pred,
                                                 y_prob=y_prob,
                                                 confidence_level=c_i,
                                                 metric_type=metric_type)
        remainigs.append(remainig)
        scores.append(score)

    plt.figure()
    plt.title("(" + metric_type + " & Remainig % of the Sample) VS. Confidence Level for " + Class + " class.")
    plt.xlabel("Confidence Level")
    plt.ylabel(metric_type + " & Remainig % of the Sample")

    plt.grid(True)

    plt.plot(confidences, remainigs, 'ro-', label="Remainig % of the Sample")
    plt.plot(confidences, scores, 'bo-', label=metric_type)

    plt.legend(loc="best")
    plt.show()

    return plt


def plot_confusion_matrices(clf, x_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix

    np.set_printoptions(precision=2)
    class_names = clf.classes_

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, Without normalization", None),
                      ("Normalized Confusion Matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, x_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize,
                                     xticks_rotation="vertical")
        disp.ax_.set_title(title)

    plt.show()
    return


def plot_confusion_matrix_wo_clf(y_true, y_pred, normalize):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    labels = list(set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap='Blues')

    return


def plot_confusion_matrix_with_confidence_level(clf, x_test, y_test, confidence_level=0.5):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    y_pred = clf.predict(x_test)
    probabilities = clf.predict_proba(x_test)
    y_prob = [max(arr_i) for arr_i in probabilities]

    y_true_post, y_pred_post = get_post_processed_arrays(y_true=y_test, y_pred=y_pred,
                                                         y_prob=y_prob, confidence_level=confidence_level)

    cm = confusion_matrix(y_true_post, y_pred_post, labels=clf.classes_, normalize="true")

    r, c = cm.shape
    for i in range(r):
        for j in range(c):
            cm[i][j] = round(cm[i][j], 2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(include_values=True, cmap='Blues', xticks_rotation='vertical')

    return


def get_highly_correlated_features(df, threshold=0.70):
    df_corr = df.corr()
    col_names = df.columns
    ncol = len(col_names)
    hcor_cols = []

    for i in range(ncol - 1):
        for j in range(i + 1, ncol):
            if abs(df_corr.iloc[i, j]) > threshold:
                hcor_cols.append(col_names[j])

    hcor_cols = list(set(hcor_cols))

    return hcor_cols


def generate_classes(y, bins):
    bins = sorted(bins)
    b = bins
    for i, x in enumerate(bins):
        if x == max(bins):
            b[i] = x * 1.0001

    classes = []
    for z in y:
        if (min(bins) <= z) and (z <= max(bins)):

            for i in range(len(b) - 1):

                if (b[i] <= z) and (z < b[i + 1]):
                    classes.append(i)
                    continue
        else:
            classes.append(-1)

    return classes


def print_regression_scores(clf, x_train, y_train, x_test, y_test):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    print('MSE on Train: %.3f' % mean_squared_error(y_train, y_train_pred))
    print('MSE on Test: %.3f' % mean_squared_error(y_test, y_test_pred))
    print()
    print('MAE on Train: %.3f' % mean_absolute_error(y_train, y_train_pred))
    print('MAE on Test: %.3f' % mean_absolute_error(y_test, y_test_pred))

    return


def plot_confusion_matrix_wo_clf(y_true, y_pred, normalize):
    labels = list(set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap='Blues')

    return


def get_bin_list(z_list, c_list, n_samples):
    bin_list = []
    for i, z in enumerate(z_list):
        if i == 0:
            count = 0
            bin_list.append(z)

        if count > n_samples:
            count = c_list[i]
            bin_list.append(z)
        else:
            count = count + c_list[i]

    return bin_list


def get_regression_scores(df_reg_results):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    df = df_reg_results.copy()
    class_list = sorted(df.test_true_classes.unique())

    scores = []
    for c_i in class_list:
        df_tmp = df[df.test_true_classes == c_i].copy()
        y_tst_tmp = df_tmp.test_true.values
        y_tst_tmp_pred = df_tmp.test_pred.values

        msr_tmp = mean_squared_error(y_tst_tmp, y_tst_tmp_pred)
        mae_tmp = mean_absolute_error(y_tst_tmp, y_tst_tmp_pred)

        scores.append([c_i, msr_tmp, mae_tmp])

    df_scores = pd.DataFrame(scores, columns=['class', 'mrs', 'mae'])
    df_scores

    return df_scores


def plot_regression_scores_for_each_class(df_scores):
    import numpy as np
    import matplotlib.pyplot as plt

    df = df_scores.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    classes = df["class"].values
    mrs = df["mrs"].values
    mae = df["mae"].values

    ## necessary variables
    ind = np.arange(len(classes))
    width = 0.35

    ## the bars
    mrs_bar = ax.bar(ind, mrs, width, color='blue')
    mae_bar = ax.bar(ind + width, mae, width, color="red")

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylabel('Scores')
    ax.set_title('MSE & MAE Scores for all Classes')

    ## add a legend
    ax.legend((mrs_bar[0], mae_bar[0]), ('Mean Squared Error', 'Mean Absolute Error'))

    plt.show()
    return


def get_shaply_ranked_features(shap_values, df_train):
    shap_means = np.abs(shap_values).mean(0)
    df_feature_importance = pd.DataFrame(list(zip(df_train.columns, shap_means)),
                                         columns=['feature_name', 'feature_importance_values'])

    df_feature_importance.sort_values(by=['feature_importance_values'],
                                      ascending=False,
                                      inplace=True)
    return df_feature_importance


def plot_elbow_check(df, df_type="pandas", k_range=range(2, 15), figsize=(12, 10)):
    import matplotlib.pyplot as plt

    sum_of_squared_distances = []

    if df_type == "pandas":
        from sklearn.cluster import KMeans

        for k in k_range:
            km = KMeans(n_clusters=k)
            km = km.fit(df)
            sum_of_squared_distances.append(km.inertia_)

    if df_type == "spark":
        from pyspark.ml.clustering import KMeans

        for k in k_range:
            km = KMeans(k=k)
            km = km.fit(df)
            sum_of_squared_distances.append(km.computeCost(df))

    plt.figure(figsize=figsize)
    plt.plot(k_range, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    return
