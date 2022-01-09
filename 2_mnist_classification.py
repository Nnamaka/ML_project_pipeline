
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


# NOTE:
# The training set is already shuffled for us, which is good because this
# guarantees that all cross-validation folds will be similar (you don’t want
# one fold to be missing some digits). Moreover, some learning algorithms
# are sensitive to the order of the training instances, and they perform poorly
# if they get many similar instances in a row. Shuffling the dataset ensures
# that this won’t happen



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
# PRECISION and RECALL
# The confusion matrix gives you a lot of information, but sometimes you
# may prefer a more concise metric
#
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

# It is often convenient to combine precision and recall into a single metric
# called the F  score, in particular if you need a simple way to compare two
# classifiers
# to compute the F1 score, simply call the f1_score() function

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

#----------------------------------------------------------------
# Precision/Recall Trade-off
# by adjusting the threshold you change the precision/Recall trade
# off
# Lowering the threshold increases recall and reduces precision
# to adjust threshold you call the "decision_function()" of the 
# classifier, which returns a score for each instance, and then use
# any threshold you want to make predictions based on those scores

y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores > threshold) # outputs True
# because threshold is low or 0

#----------------------------------------------------------------
threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred # outputs False because the threshold is high

# How do you decide the threshold to use? first use the 
#"cross_val_predict()" function to get the scores of all instances
# in the training seet, but this time specify that you want to 
# return decision scores instead of predictions

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                            method="decision_function")

# then with these scores, use the "precision_recall_curve()" 
# function to compute precision and recall for all possible 
# threshold
    
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5,
y_scores)

#-----------------------------------------------------------
# Another way to select a good precision/recall trade-off is to
# plot precision directly against recall
# Suppose you decide to aim for 90% precision. You look up the first plot
# and find that you need to use a threshold of about 8,000. To be more
# precise you can search for the lowest threshold that gives you at least 90%
# precision (np.argmax() will give you the first index of the maximum
# value, which in this case means the first True value):

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# To make predicitions(for now ), instead of calling the classi
# -fier's predict() method, you can run this code:

y_train_pred_90 = (y_scores >= threshold_90_precision)


#--------------------------------------------------------------
# The ROC Curve
# The receiver operating characteristic (ROC) curve is another common
# tool used with binary classifiers. It is very similar to the precision/recall
# curve, but instead of plotting precision versus recall, the ROC curve plots
# the true positive rate (another name for recall) against the false positive
# rate (FPR)

# To plot the ROC curve, you first use the roc_curve() function to
# compute the TPR and FPR for various threshold values

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


#--------------------------------------------------------------
# MULTICLASS CLASSIFICATION
#