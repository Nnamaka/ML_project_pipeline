
# One of the most important transformations you need to apply
# to your data is "feature scaling". with few exceptions, Machine
# learning algorithms don't perform well when the input numerical
# attdributes have very different scales. This is the case for 
# the housing data: the total numbers of rooms ranges from about 6
# 39,320, while the median incomes only range from 0 to 15. Note
# that scaling the target values is generally not required.

# 
# There are two common ways to get all attributes to have the same scale: min-
# max scaling and standardization.

# Min-max scaling(also called normalization) is the simplest: values
# are shifted and rescaled so that they end up ranging from 0 to 1.
#
# Unlike min-max scaling,
# standardization does not bound values to a specific range, which may be a
# problem for some algorithms (e.g., neural networks often expect an input
# value ranging from 0 to 1).
#-------------------------------------------------------------

# Scikit-Learn provides a transformer called "MinMaxScaler" for this. It has a
# feature_range hyperparameter that lets you change the range if, for some
# reason, you don’t want 0–1.
#
# Scikit-Learn provides a transformer called "StandardScaler" for
# standardization.

#----------------------------------------------------------------
# As with all the transformations, it is important to fit the scalers to the training data only,
# not to the full dataset (including the test set). Only then can you use them to transform the
# training set and the test set (and new data).



