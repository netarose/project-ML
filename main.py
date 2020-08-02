import time
import pandas as pd
import numpy as np
import os

from xbart import XBART
from kigb.core.lgbm.lkigb import LKiGB as KiGB
from mercs.core import Mercs
import xgboost

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

'''
wrapper for XBART class
get_params and set_params were added to use the RandomizedSearchCV
'''
class xbart_(XBART):

    def __init__(self, model_num=0, num_trees: int = 100, num_sweeps: int = 40, n_min: int = 1,
                 num_cutpoints: int = 100, alpha: float = 0.95, beta: float = 1.25, tau="auto",
                 burnin: int = 15, mtry="auto", max_depth_num: int = 250,
                 kap: float = 16.0, s: float = 4.0, verbose: bool = False,
                 parallel: bool = False, seed: int = 0, model: str = "Normal",
                 no_split_penality="auto", sample_weights_flag: bool = True, num_classes=1):
        super(xbart_, self).__init__(num_trees, num_sweeps, n_min, num_cutpoints, alpha, beta, tau,
                                     burnin, mtry, max_depth_num, kap, s, verbose, parallel, seed, model,
                                     no_split_penality, sample_weights_flag, num_classes)

    def get_params(self, deep):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self

###############################################################################################################

'''
wrapper for Mercs class
set_params was added to use the RandomizedSearchCV
'''
class mercs_w(Mercs):

    def __init__(self, selection_algorithm="base",
        induction_algorithm="base",
        classifier_algorithm="DT",
        regressor_algorithm="DT",
        prediction_algorithm="mi",
        inference_algorithm="own",
        imputer_algorithm="default",
        evaluation_algorithm="dummy",
        random_state=42,
        **kwargs
    ):
        super(mercs_w, self).__init__(selection_algorithm, induction_algorithm, classifier_algorithm,
                                      regressor_algorithm, prediction_algorithm, inference_algorithm, imputer_algorithm,
                                      evaluation_algorithm, random_state, **kwargs)

    def set_params(self, **params):
        self.params = params
        return self

###############################################################################################################

# optimize the KiGB hyper-parameters
def optimize_params_KiGB(reg):
    # Set up possible values of parameters to optimize over
    params = dict()
    params["trees"] = [10, 20, 30, 40]
    params["learning_rate"] = [0.005, 0.01, 0.05, 0.1]
    params["max_depth"] = [3, 6, 10, -1]

    # Define inner folds fo parameters optimization
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

    scoring = {'r2': 'r2'}
    model = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring=scoring, cv=inner_cv, verbose=1,
                               refit='r2', n_jobs=1, n_iter=50)
    return model

# optimize the xbart hyper-parameters
def optimize_params_xbart(reg):
    from sklearn.model_selection import RandomizedSearchCV

    # Set up possible values of parameters to optimize over
    params = dict()
    params["num_trees"] = [30, 50, 100, 150]
    params["num_sweeps"] = [21, 30, 40, 60]
    params["burnin"] = [5, 10, 15, 20]

    # Define inner folds fo parameters optimization
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

    # Non_nested parameter search and scoring
    scoring = {'r2': 'r2'}
    model = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring=scoring, cv=inner_cv, verbose=1,
                               refit='r2', n_jobs=1)
    return model

# optimize the MERCS hyper-parameters
def optimize_params_MERCS(reg):
    # Set up possible values of parameters to optimize over
    params = dict()
    params["nb_iterations"] = [2, 5, 8, 10]
    params["regressor_algorithm"] = ['xgb', 'DT', 'RF', 'lgbm']
    params["max_depth"] = [3, 6, 9, 12]

    # Define inner folds fo parameters optimization
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

    scoring = {'r2': 'r2'}
    model = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring=scoring, cv=inner_cv, verbose=1,
                               refit='r2', n_iter=50) #, n_jobs=1
    return model

# optimize the XGBoost hyper-parameters
def optimize_params_xgb(reg):
    # Set up possible values of parameters to optimize over
    params = dict()
    params["n_estimators"] = [15, 10, 8, 5]
    params["learning_rate"] = [0.005, 0.01, 0.05, 0.1]
    params["max_depth"] = [3, 6, 9, 12]

    # Define inner folds fo parameters optimization
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)

    scoring = {'r2': 'r2'}
    model = RandomizedSearchCV(estimator=reg, param_distributions=params, scoring=scoring, cv=inner_cv, verbose=1,
                               refit='r2', n_jobs=1, n_iter=50)
    return model

# predict evaluation metrics of the model
def pred_metrics(y_val, y_pred, metrics_info_dict):
    metrics_info_dict["MeanSE"] = metrics.mean_squared_error(y_val, y_pred)
    metrics_info_dict["MeanAE"] = metrics.mean_absolute_error(y_val, y_pred)
    metrics_info_dict["MedianAE"] = metrics.median_absolute_error(y_val, y_pred)
    metrics_info_dict["r2"] = metrics.r2_score(y_val, y_pred)
    metrics_info_dict["EVS"] = metrics.explained_variance_score(y_val, y_pred)

    return metrics_info_dict

# convert categorical attributes to numeric attributes
def categorical_to_numeric(data):
    first_row = data.iloc[0]
    for index in range(len(first_row)):
        val = first_row[index]
        if type(val) is str:
            col = data.columns[index]
            data[col] = pd.factorize(data[col])[0]

    return data

def KiGB_model():

    results_dict = []
    for index, fn in enumerate(os.listdir('data')):
        data = pd.read_csv('data/' + fn)
        data = categorical_to_numeric(data)
        data = data.fillna(data.median())        # fill missing values with column median
        class_name = data.columns[-1]
        X = data.drop(class_name, axis=1)
        y = data[class_name]
        #y = (y - y.min()) / (y.max() - y.min())  # normalize target column between 0 to 1
        kf = KFold(n_splits=10, shuffle=True)
        kf_index = 1
        for train_index, test_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            '''Train the model'''
            advice  = np.array([1], dtype=int)
            kigb = KiGB(lamda=1, epsilon=0.1, advice=advice, objective='regression', trees=30)
            kigb = optimize_params_KiGB(kigb)
            fit_time = time.time()
            kigb.fit(X_train, y_train)
            fit_time = (time.time() - fit_time)*1000

            '''Test the model'''
            pred_time = time.time()
            y_pred = kigb.predict(X_val)
            pred_time = (time.time() - pred_time) * (1000/len(X_val)) * 1000

            '''Calculate evaluation metrics'''
            metrics_info_dict = {}
            metrics_info_dict = pred_metrics(y_val, y_pred, metrics_info_dict)
            results_dict.append([fn, "KiGB", kf_index,
                                 "trees:" + str(kigb.best_estimator_.get_params()['trees']) +
                                 " learning_rate:" + str(kigb.best_estimator_.get_params()['learning_rate']) +
                                 " max_depth:" + str(kigb.best_estimator_.get_params()['max_depth']),
                                 metrics_info_dict["MeanSE"], metrics_info_dict["MeanAE"],
                                 metrics_info_dict["MedianAE"], metrics_info_dict["r2"],
                                 metrics_info_dict["EVS"], fit_time, pred_time])
            kf_index += 1

    '''Save results to csv file'''
    results = pd.DataFrame(results_dict)
    results.columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper Parameters',
                       'Mean Squared Error', 'Mean Absolute Error', 'Median Absolute Error',
                       'r2 Score', 'Explained Variance Score', 'Training Time', 'Inference Time']
    results.to_csv('KiGB_results.csv', index=False)

def XBART_model():
    results_dict = []
    for index, fn in enumerate(os.listdir('data')):
        data = pd.read_csv('data/' + fn)
        data = categorical_to_numeric(data)
        data = data.fillna(data.median())        # fill missing values with column median
        class_name = data.columns[-1]
        X = data.drop(class_name, axis=1)
        y = data[class_name]
        #y = (y - y.min()) / (y.max() - y.min())  # normalize target column between 0 to 1
        kf = KFold(n_splits=10, shuffle=True)
        kf_index = 1
        for train_index, test_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            '''Train the model'''
            xbt = xbart_(num_trees=100, num_sweeps=40, burnin=15)
            xbt = optimize_params_xbart(xbt)

            fit_time = time.time()
            xbt.fit(X_train, y_train)
            fit_time = (time.time() - fit_time) * 1000

            pred_time = time.time()
            y_pred = xbt.predict(X_val)  # Return n X num_sweeps matrix
            pred_time = (time.time() - pred_time) * (1000 / len(X_val)) * 1000

            '''Calculate evaluation metrics'''
            metrics_info_dict = {}
            metrics_info_dict = pred_metrics(y_val, y_pred, metrics_info_dict)
            results_dict.append([fn, "xbart", kf_index,
                                 "num_trees:" + str(xbt.best_estimator_.get_params(False)['num_trees']) +
                                 " num_sweeps:" + str(xbt.best_estimator_.get_params(False)['num_sweeps']) +
                                 " burnin:" + str(xbt.best_estimator_.get_params(False)['burnin']),
                                 metrics_info_dict["MeanSE"], metrics_info_dict["MeanAE"],
                                 metrics_info_dict["MedianAE"], metrics_info_dict["r2"],
                                 metrics_info_dict["EVS"], fit_time, pred_time])
            kf_index += 1

    '''Save results to csv file'''
    results = pd.DataFrame(results_dict)
    results.columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper Parameters',
                       'Mean Squared Error', 'Mean Absolute Error', 'Median Absolute Error',
                       'r2 Score', 'Explained Variance Score', 'Training Time', 'Inference Time']
    results.to_csv('XBART_results.csv', index=False)

def MERCS_model():
    results_dict = []
    for index, fn in enumerate(os.listdir('data')):
        data = pd.read_csv('data/' + fn)
        data = categorical_to_numeric(data)
        data = data.fillna(data.median())        # fill missing values with column median
        class_name = data.columns[-1]
        X = data.drop(class_name, axis=1)
        y = data[class_name]
        y = (y - y.min()) / (y.max() - y.min())  # normalize target column between 0 to 1
        kf = KFold(n_splits=10, shuffle=True)
        kf_index = 1
        for train_index, test_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            X_train = np.array(X_train)
            X_val = np.array(X_val)
            y_train = np.array(y_train)

            '''Train the model'''
            if fn == "detroit.csv" or fn == "longley.csv":
                mrc = mercs_w(max_depth=4, selection_algorithm="random", fraction_missing=0.6, nb_targets=1, nb_iterations=2,
                            verbose=1, inference_algorithm="own", max_steps=8, prediction_algorithm="it", evaluation_algorithm="base")
            else:
                mrc = mercs_w(max_depth=4, selection_algorithm="random", fraction_missing=0.6, nb_targets=1, nb_iterations=2,
                            verbose=1, inference_algorithm="own", max_steps=8, prediction_algorithm="it")
            mrc = optimize_params_MERCS(mrc)
            fit_time = time.time()
            mrc.fit(X_train, y=y_train, nominal_attributes={0})
            fit_time = (time.time() - fit_time) * 1000

            '''Test the model'''
            pred_time = time.time()
            y_pred = mrc.predict(X_val)
            pred_time = (time.time() - pred_time) * (1000 / len(X_val)) * 1000

            '''Calculate evaluation metrics'''
            metrics_info_dict = {}
            metrics_info_dict = pred_metrics(y_val, y_pred, metrics_info_dict)
            results_dict.append([fn, "MERCS", kf_index,
                                 "nb_iterations:" + str(mrc.best_estimator_.get_params()['nb_iterations']) +
                                 " regressor_algorithm:" + str(mrc.best_estimator_.get_params()['regressor_algorithm']) +
                                 " max_depth:" + str(mrc.best_estimator_.get_params()['max_depth']),
                                 metrics_info_dict["MeanSE"], metrics_info_dict["MeanAE"],
                                 metrics_info_dict["MedianAE"], metrics_info_dict["r2"],
                                 metrics_info_dict["EVS"], fit_time, pred_time])
            kf_index += 1

    '''Save results to csv file'''
    results = pd.DataFrame(results_dict)
    results.columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper Parameters',
                       'Mean Squared Error', 'Mean Absolute Error', 'Median Absolute Error',
                       'r2 Score', 'Explained Variance Score', 'Training Time', 'Inference Time']
    results.to_csv('MERCS_results.csv', index=False)

def XGB_model():
    results_dict = []
    for index, fn in enumerate(os.listdir('data')):
        data = pd.read_csv('data/' + fn)
        data = categorical_to_numeric(data)
        data = data.fillna(data.median())        # fill missing values with column median
        class_name = data.columns[-1]
        X = data.drop(class_name, axis=1)
        y = data[class_name]
        #y = (y - y.min()) / (y.max() - y.min())  # normalize target column between 0 to 1
        kf = KFold(n_splits=10, shuffle=True)
        kf_index = 1
        for train_index, test_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            '''Train the model'''
            xgb = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.01,
                 max_depth=3,
                 n_estimators=10,
                 subsample=0.6)
            xgb = optimize_params_xgb(xgb)
            fit_time = time.time()
            xgb.fit(X_train, y_train)
            fit_time = (time.time() - fit_time) * 1000

            '''Test the model'''
            pred_time = time.time()
            y_pred = xgb.predict(X_val)
            pred_time = (time.time() - pred_time) * (1000 / len(X_val)) * 1000

            '''Calculate evaluation metrics'''
            metrics_info_dict = {}
            metrics_info_dict = pred_metrics(y_val, y_pred, metrics_info_dict)
            results_dict.append([fn, "XGBoost", kf_index,
                                 "n_estimators:" + str(xgb.best_estimator_.get_params()['n_estimators']) +
                                 " learning_rate:" + str(xgb.best_estimator_.get_params()['learning_rate']) +
                                 " max_depth:" + str(xgb.best_estimator_.get_params()['max_depth']),
                                 metrics_info_dict["MeanSE"], metrics_info_dict["MeanAE"],
                                 metrics_info_dict["MedianAE"], metrics_info_dict["r2"],
                                 metrics_info_dict["EVS"], fit_time, pred_time])
            kf_index += 1

    '''Save results to csv file'''
    results = pd.DataFrame(results_dict)
    results.columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation', 'Hyper Parameters',
                       'Mean Squared Error', 'Mean Absolute Error', 'Median Absolute Error',
                       'r2 Score', 'Explained Variance Score', 'Training Time', 'Inference Time']
    results.to_csv('XGBoost_results.csv', index=False)


def main():
    start = time.time()

    KiGB_model()        #scikit-learn 0.23.1
    XBART_model()       #scikit-learn 0.23.1
    MERCS_model()
    XGB_model()
    #print("run time: " + str((time.time() - start)))

if __name__ == '__main__':
    main()
