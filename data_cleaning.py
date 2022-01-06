
# most machine learning algorithms cannot work with missing
# features, so lets create a few functions to take care of them.
#--------------------------------------------------------------

# we saw earlier that "total_bedrooms" attribute has some missing
# values, so let's fix this.
# so you have three options
# 1. Get rid of the corresponding districts
# 2. Get rid of the whole attribute
# 3. Set the values to some value(zero, the mean, the median, etc)
# all this three options can be accomplished using DataFrame's
# "dropna()", "drop()" , and "fillna()"

housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

# If you choose option 3, you should compute the median value on the training
# set and use it to fill the missing values in the training set. Don’t forget to save
# the median value that you have computed. You will need it later to replace
# missing values in the test set when you want to evaluate your system, and also
# once the system goes live to replace missing values in new data.

#-----------------------------------------------------------------
# SCIKIT-LEARN provides a handy class to take care of missing 
# values: "SimpleImputer". to use it, first you need to create a
# "SimpleImputer" instance, specifying that you want to replace
# each attribute's missing values with the median of that attribute

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# since the median can only be computed on nemerical attributes
# you need to create a copy of the datawithout the text attribute
# "ocean_porximity"

housing_num = housing.drop("ocean_proximity", axis=1)

# now you can fit the imputer instance to the training data using
# the fit() method

imputer.fit(housing_num)

# The imputer has simply computed the median of each attribute and stored the
# result in its statistics_ instance variable. Only the total_bedrooms
# attribute had missing values, but we cannot be sure that there won’t be any
# missing values in new data after the system goes live, so it is safer to apply
# the imputer to all the numerical attributes:


imputer.statistics_
# or 
housing_num_median().values

# Now you can use this "trained" imputer to transform the training
# set by replacing missing values with the learned medians:

X = imputer.transform(housing_num)

# The result is a plain Numpy array containing the transformed 
# features. if you want to put it bact into a pandas Dataframe:

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
index=housing_num.index)

