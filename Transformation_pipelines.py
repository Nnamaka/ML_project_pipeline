
# As you can see, there are many data transformation steps that
# need to be executed in the right order. fortunately, Scikit-learn
# provides the pipeline class to help with such sequences of 
# transformations. Here is a small pipeline for the numerical
# attributes

from sklearn.pipeline import pipeline
from sklearn.preprocessing import standardScaler

num_pipeline = pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', standardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# The Pipeline constructor takes a list of name/estimator pairs defining a
# sequence of steps. All but the last estimator must be transformers (i.e., they
# must have a fit_transform() method). The names can be anything you like
# (as long as they are unique and don’t contain double underscores, __); they
# will come in handy later for hyperparameter tuning.
# When you call the pipeline’s fit() method, it calls fit_transform()
# sequentially on all transformers, passing the output of each call as the
# parameter to the next call until it reaches the final estimator, for which it calls
# the fit() method.
# The pipeline exposes the same methods as the final estimator. In this example,
# the last estimator is a StandardScaler, which is a transformer, so the
# pipeline has a transform() method that applies all the transforms to the data
# in sequence (and of course also a fit_transform() method, which is the one
# we used).

# So far, we have handled the categorical columns and the numerical columns
# separately. It would be more convenient to have a single transformer able to
# handle all columns, applying the appropriate transformations to each column.
# In version 0.20, Scikit-Learn introduced the ColumnTransformer for this
# purpose, and the good news is that it works great with pandas DataFrames.
# Let’s use it to apply all the transformations to the housing data:

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)


# First we import the ColumnTransformer class, next we get the list of
# numerical column names and the list of categorical column names, and then
# we construct a ColumnTransformer. The constructor requires a list of tuples,
# where each tuple contains a name,  a transformer, and a list of names (or
# indices) of columns that the transformer should be applied to. In this example,
# we specify that the numerical columns should be transformed using the
# num_pipeline that we defined earlier, and the categorical columns should be
# transformed using a OneHotEncoder. Finally, we apply this
# ColumnTransformer to the housing data: it applies each transformer to the
# appropriate columns and concatenates the outputs along the second axis (the
# transformers must return the same number of rows).


# Note that the OneHotEncoder returns a sparse matrix, while the
# num_pipeline returns a dense matrix. When there is such a mix of sparse and
# dense matrices, the ColumnTransformer estimates the density of the final
# matrix (i.e., the ratio of nonzero cells), and it returns a sparse matrix if the
# density is lower than a given threshold (by default, sparse_threshold=0.3).
# In this example, it returns a dense matrix. And that’s it! We have a
# preprocessing pipeline that takes the full housing data and applies the
# appropriate transformations to each column.


