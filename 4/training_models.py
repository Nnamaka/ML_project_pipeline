
# The benefits of learning whats under the hood and having a good
# understanding of how things work can help you:
# 1. quickly home in on the appropriate model
# 2. use the right training algorithm to use, and a good set of 
# hyperparameters for your task.
# 3. Debug issues and perform error analysis more efficiently.
#---------------------------------------------------------------
# here we discuss two different ways to train linear regression
# model
# a) using a direct "closed-form" equation that directly computes the
# model parameters that best fit the model to the training set(ie
# the model parameters that minimize the cost function over the
# training set)
# b) using an itereative optimization approach called gradient
# descent that gradually tweaks the model parameters to minimize
# the cost function over the training set, eventually converging
# to the same set of parameters as the first method. there are 
# a few variants of gradient descent: Batch GD, Mini-batch GD
# and Stochastic GD
#
# we will also disccuss polynomial regression, a more complex model
# that can fit nonlinear datasets. this model can easily overfit
# the training data, so we will look at how to detect whether or 
# not it is overfitting the train data using learning curves, and
# then we will look at several regularization techniques that can
# reduce the rist of overfitting that training set

# we looked at a simple regression model of life satisfaction:
# life_satisfaction = b + m x GDP_per_capita
#
# basically the model above is just a linear function of the input
# feature GDP_per_capita. b and m are the model parameters.

# more generally, a linear model makes a prefiction by simply
# computing a weighted sum of the inputs features, plus a constant
# called th bias term
#
# but how do we train it? Well, recall
# that training a model means setting its parameters so that the model best fits
# the training set
# let's generate some linear-looking data to test this equation

import numpy as np

x = 2 * np.random.rand(100, 1)

# the function that we used to generate the data is y = 4 + 3x +
# gaussian noise
y = 4 + 3 * x + np.random.randn(100,1)

x_b = np.c_[np.ones((100, 1)), x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

theta_best

# we would have hoped for 4 and 3, but the noise made it impossible
# to recover the exact parameters of the original function.

# now to make predictions 

x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2,1)), x_new]
y_predict = x_new_b.dot(theta_best)
y_predict


#----------------------------------------------------------------
# Gradient Descent
# Gradient descent measures the local gradient of the error function
# with regard to the parameter vector, and it goes in the direction
# of the descending gradient. Once the gradient is zero, you have
# reached a minimum

# concretely, you start by filling theta with random values (this is 
# called random initialization). Then you improve it gradually, taking
# one baby step at a time, each step attempting to decrease the 
# cost function (e.g the MSE), until the algorithm converges to a
# minimum

# An important parameter in gradient descent is the size of the steps,
# determined by the learning rate hyperparameter. if the learning
# rate is too small, then the algorithm will have to go through
# many iterations to converge, which will take a long time
# on the other hand, if the learning rate is too high, you might
# jump across the valley and end up on the other side, possibly
# even higher up than you were before. This might make the 
# algorithm diverge, with larger and larger values, failing to 
# find a good solution

# NOTE: when using gradient descent, you should ensure that all 
# features have a similar scale (e.g using Scikit-learn's 
# StandardScaler class), or else it will take much longer to converge

# Batch Gradient descent uses the whole batch of training data at
# every step( actually, full gradient descent would probably be a
# a better name)

# -----------------------------------------------------------------
# Polynomial Regression
# if your data is more complex than a straight line? surprisingly
# you can use a linear model to fit nonlinear data. A simple way 
# to do this is to add powers of each feature as new features, then
# train a linear model on this extended set of features.
# see example below. 'X' is the data:

from sklearn.preprocessing import PolynomialFea🇹🇲 
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
x_poly[0]

# now we can fit a "linearRegression" model to this extended training
# data

lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
lin_reg.intercept_, lin_reg.coef_

#-----------------------------------------------------------------
# UNDERFITTING AND OVERFITTING
# how can you tell that your model is overfitting or underfitting
# the data?
# a) use cross-validation to get an estimate of a model's 
# generalization performance.
# b) use learning curves: view the performance of your model for 
# training set and validation set on plots
# see the following code to plot curves:

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],
y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

# now to look at the learning curves
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# TIPS
# if your model is underfitting the training data, adding more 
# training examples will not help. You need to use a more complex
# model or come up with better features
# one way to improve an overfitting model is to feed it more 
# training data until the validation error reaches the training
# error

#---------------------------------------------------------------
# REGULARIZED LINEAR MODELS
# a good way to reduce overfitting is to regularize the model(i.e
# to constrain it)
# a good way to regularize a polynomial model is to reduce the
# number of polynomial degrees.
# for a linear model, regularization is typically achieved by 
# constraining the weights of the model.
# Ridge Regression, Lasso Regression, and Elastic Net, are three
# different ways to constrain the weights

# Note: it is important to scale the data( e.g using a "StandardScaler")
# before performing Ridge Regression, as it is sensitive to the 
# scale of the input features. This is true of most regularized models
