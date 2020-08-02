import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from xgboost import DMatrix


def get_feature_importance(booster, type):
    fi = booster.get_score(importance_type=type)
    all_features = [fi.get(f, 0.) for f in booster.feature_names]
    all_features = np.array(all_features, dtype=np.float32)
    return all_features / all_features.sum()

def train_meta_learner():
    data = pd.read_csv('meta_data.csv')
    data = data.drop('name', axis=1)
    X = data.drop('algorithm win', axis=1)
    X['algorithm'] = pd.factorize(X['algorithm'])[0]   # convert algorithm name attribute to numeric attribute
    y = data['algorithm win']
    feature_importance = []
    results_dict = []

    kf = KFold(n_splits=100, shuffle=False)
    kf_index = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = xgb.XGBClassifier(max_depth=6, n_estimators=100, eta=0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        # metrics calculation
        ############################################################
        cm = metrics.confusion_matrix(y_test, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        FP = 0
        FN = 0
        TP = 0
        TN = 0
        i = 0
        for Yt in y_test:
            if Yt == 1:
                if y_pred[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if y_pred[i] == 1:
                    FP += 1
                else:
                    TN += 1
            i += 1

        accuracy = metrics.accuracy_score(y_test, y_pred)
        if TP == 0:
            precision = 0
            TPR = 0
        else:
            precision = TP/(TP + FP)
            TPR = TP/(TP + FN)
        if FP == 0:
            FPR = 0
        else:
            FPR = FP/(TN + FP)
        auc = metrics.roc_auc_score(y_test, y_pred_proba[:, 1])
        pr_curve = metrics.average_precision_score(y_test, y_pred_proba[:, 1], average="micro")

        results_dict.append([kf_index, accuracy, TPR, FPR, precision, auc, pr_curve])
        kf_index += 1

    results = pd.DataFrame(results_dict)
    results.columns = ['Cross Validation', 'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR_curve']
    results.to_csv('meta_results.csv', index=False)

    # feature importance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = xgb.XGBClassifier(max_depth=6, n_estimators=100, eta=0.2)
    clf.fit(X_train, y_train)
    booster = clf.get_booster()
    feature_importance.append(get_feature_importance(booster, 'gain'))
    feature_importance.append(get_feature_importance(booster, 'cover'))
    feature_importance.append(get_feature_importance(booster, 'weight'))
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance.columns = [X.columns]
    feature_importance.to_csv('feature_importance.csv', index=False)

    booster = clf.get_booster()
    shap = booster.predict(DMatrix(X_test), pred_contribs=True)
    np.savetxt("shap.csv", shap, delimiter=",")

def main():
    start = time.time()
    train_meta_learner()
    print("run time: " + str((time.time() - start)))

if __name__ == '__main__':
    main()
