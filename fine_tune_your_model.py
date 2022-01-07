
# let's assume that you have a shortlist of promising models. you
# now need to fine-tune them. below are a few ways to do that
# -------------------------------------------------------------
# Grid Search

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimstors': [3, 10 , 30], 'max_features': [2, 4, 6, 3]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2
    , 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                        scoring='neg_mean_squared_error',
                        return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)


# This param_grid tells Scikit-Learn to first evaluate all 3 × 4 = 12
# combinations of n_estimators and max_features hyperparameter values
# specified in the first dict, then try all 2 × 3 = 6
# combinations of hyperparameter values in the second dict, but this time with
# the bootstrap hyperparameter set to False instead of True (which is the
# default value for this hyperparameter).
#
# The grid search will explore 12 + 6 = 18 combinations of
# RandomForestRegressor hyperparameter values, and it will train each model
# 5 times (since we are using five-fold cross validation). In other words, all in
# all, there will be 18 × 5 = 90 rounds of training! It may take quite a long time,
# but when it is done you can get the best combination of parameters like this:

grid_search.best_params_


# you can also get the best estimator directly
grid_search.best_estimator


# If GridSearchCV is initialized with refit=True (which is the default), then once it finds
# the best estimator using cross-validation, it retrains it on the whole training set. This is
# usually a good idea, since feeding it more data will likely improve its performance
# ----------------------------------------------------------------
# Don’t forget that you can treat some of the data preparation steps as hyperparameters. For
# example, the grid search will automatically find out whether or not to add a feature you
# were not sure about (e.g., using the add_bedrooms_per_room hyperparameter of your
# CombinedAttributesAdder transformer). It may similarly be used to automatically find
# the best way to handle outliers, missing features, feature selection, and more.


# Another way to fine-tune your system is to try to combine the models that
# perform best. The group (or “ensemble”) will often perform better than the
# best individual model (just like Random Forests perform better than the
# individual Decision Trees they rely on), especially if the individual models
# make very different types of errors.

