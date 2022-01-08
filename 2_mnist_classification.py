
# scikit-learn provides many helper functions to download popular
# datasets. MNIST is one of them. The following code fetches the
# MNIST dataset

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# we use the "keys()" function to see the features of the dataset
# containded in a dictionary structure
mnist.keys()

# There are 70,000 images, and each image has 784 features. This is because
# each image is 28 × 28 pixels, and each feature simply represents one
# pixel’s intensity, from 0 (white) to 255 (black). Let’s take a peek at one
# digit from the dataset. All you need to do is grab an instance’s feature
# vector, reshape it to a 28 × 28 array, and display it using Matplotlib’s
# imshow() function:

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

#lets look at the corresponding label for the first digit we just saw
y[0]

# note that the label is a string. Most ML algorithms expect 
# numbers, so let's cast "y" to integer
y = y.astype(np.uint8)


# you should always create a test set and set it aside before 
# inspecting the data closely. Though the dataset has been splited 
# already. But nevertheless we will split things further one more 
# time

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],
y[60000:]

#---------------------------------------------------------------
# BINARY CLASSIFIER
# we simplify the problem for now and only try to indentify one
# digit- the number 5. ("5-detector") will be an example of a 
# binary classifier capable of distinguishing between two classes
# 5 and not-5
# lets create the target for this classification task:

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Now let’s pick a classifier and train it. A good place to start is with a
# Stochastic Gradient Descent (SGD) classifier, using Scikit-Learn’s
# SGDClassifier class. This classifier has the advantage of being capable
# of handling very large datasets efficiently. This is in part because SGD
# deals with training instances independently, one at a time (which also
# makes SGD well suited for online learning), as we will see later. Let’s
# create an SGDClassifier and train it on the whole training set:

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# This demonstrates why accuracy is generally not the preferred
# performance measure for classifiers, especially when you are dealing with
# skewed datasets (i.e., when some classes are much more frequent than
# others)

#----------------------------------------------------------------
# confusion_matrix:
# A much better way to evaluate the performance of a classifier is to look at
# the confusion matrix. The general idea is to count the number of times
# instances of class A are classified as class B. For example, to know the
# number of times the classifier confused images of 5s with 3s, you would
# look in the fifth row and third column of the confusion matrix.

# to compute the confusion matrix, you first need to havve a set
# of predictions so that they can be compared to the actual targets.
# we could make predictions on the test set, but let's keep it 
# untouched for now(remember that you want to use the test seet only
# at the very end of your project, once you have a classifier that
# you are ready to launch)

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Just like the cross_val_score() function, cross_val_predict()
# performs K-fold cross-validation, but instead of returning the evaluation
# scores, it returns the predictions made on each test fold.

# the confusion matrix is described below
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

# Each row in a confusion matrix represents an actual class, while each
# column represents a predicted class. The first row of this matrix considers
# non-5 images (the negative class): 53,057 of them were correctly
# classified as non-5s (they are called true negatives), while the remaining

#------------------------------------------------------------
# Another more concise metric which is
# PRECISION


