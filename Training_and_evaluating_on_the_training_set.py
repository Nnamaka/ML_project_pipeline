
# The good news is that thanks to all these previous steps, things
# are now going to be much simpler than you might think.
# lets train a Linear Regression model

from sklearn.linear_model import linearRegression

lin_reg = linearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# now lets try it out on a few instances from the training set

some_data = housing.iloc[:5] # raw training data
some_labels = housing_labels.iloc[:5]

# raw training data is transformed
some_data_prepared = full_pipeline.transform(some_data) 
print("predictions:", lin_reg.predict(some_data_prepared))

# now we measure the model RMSE( or the error)/mean squared error
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)


# the model is underfitting
# This is an example of a
# model underfitting the training data. When this happens it can mean that the
# features do not provide enough information to make good predictions, or that
# the model is not powerful enough. As we saw in the previous chapter, the
# main ways to fix underfitting are to select a more powerful model, to feed the
# training algorithm with better features, or to reduce the constraints on the
# model. This model is not regularized, which rules out the last option. 

#--------------------------------------------------------------
# lets try a more complex model to see how it does
# we will try a "DecisionTreeRegressor". This is a powerful model
# capable of finding complex nonlinear relationships in the data

from sklearn.metrics import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# now lets measure the models error
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

# !!! the model gives 0 errors. it is much more likely that the 
# model has overfit the data. to evaluate the model we use part of
# the training set and not touch the previously carved out test set

#----------------------------------------------------------------
# two ways to evaluate our model
# 1) One way to evaluate the Decision Tree model would be to use the
# train_test_split() function to split the training set into a smaller training
# set and a validation set, then train your models against the smaller training set
# and evaluate them against the validation set. It’s a bit of work, but nothing too
# difficult, and it would work fairly well.
#
# 2) A great alternative is to use Scikit-Learn’s K-fold cross-validation feature.

# we do cross-validation on the "Decision Tree model"
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt( -scores)

# Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather
# than a cost function (lower is better), so the scoring function is actually the opposite of
# the MSE (i.e., a negative value), which is why the preceding code computes -scores
# before calculating the square root.


# function to display score values
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation", scores.std())

display_scores(tree_rmse_scores)

# cross-validation allows you to get not only an estimate of the performance of your
# model, but also a measure of how precise this estimate is (i.e., its standard
# deviation). The Decision Tree has a score of approximately 71,407, generally
# ±2,439. You would not have this information if you just used one validation
# set. But cross-validation comes at the cost of training the model several times,
# so it is not always possible.

# here we do cross-validation on the "lineaer regression" model

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# now lets look at the "RandomForestRegressor" model
# the Random forests work by training many Decision Trees on
# random subsets of the features, then averaging out their predictions
# Building a model on top of many other models is called Ensemble Learning

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# doing cross-validation on the "Random forest"
from sklearn.model_selection import cross_val_score

forest_scores =cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
                            
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# Random Forests look very promising. However,
# note that the score on the training set is still much lower than on the
# validation sets, meaning that the model is still overfitting the training set.
# Possible solutions for overfitting are to simplify the model, constrain it (i.e.,
# regularize it), or get a lot more training data. Before you dive much deeper
# into Random Forests, however, you should try out many other models from
# various categories of Machine Learning algorithms (e.g., several Support
# Vector Machines with different kernels, and possibly a neural network),
# without spending too much time tweaking the hyperparameters. The goal is to
# shortlist a few (two to five) promising models.

