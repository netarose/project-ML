# project-ML
 
To run this project first insall MERCS and XBART using the commands:

	pip install mercs
	
	pip install xbart

In our experiments we used:

	keras 2.2.4
	
	tensorflow 1.13.1
	
	scikit-learn 0.23.1
	
	xgboost 1.1.1

To run the regression models run the main.py file.

The results will be saved in KiGB_results.csv, MERCS_results.csv, XBART_results.csv, XGBoost_results.csv.

To run the meta-learning model run the meta_learning_model.py file.

The results will be saved in meta_results.csv, feature_importance.csv, shap.csv.



# KiGB Hyper-parameters:

	trees = [10, 20, 30, 40]
	learning_rate = [0.005, 0.01, 0.05, 0.1]
	max_depth = [3, 6, 10, -1]

# MERCS Hyper-parameters:

    num_trees = [30, 50, 100, 150]
    num_sweeps = [21, 30, 40, 60]
    burnin = [5, 10, 15, 20]
   
# XBART Hyper-parameters:

    nb_iterations = [2, 5, 8, 10]
    regressor_algorithm = ['xgb', 'DT', 'RF', 'lgbm']
    max_depth = [3, 6, 9, 12]
    
# XGBoost Hyper-parameters:

    n_estimators = [15, 10, 8, 5]
    learning_rate = [0.005, 0.01, 0.05, 0.1]
    max_depth = [3, 6, 9, 12]
