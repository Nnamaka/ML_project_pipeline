
# now let's look at text attributes. in this dataset, there is just
# one attribute with text/categorical format: the "ocean_proximity"
# attribute.
# looking at the value of the first 10 instances

housing_cat = housing["ocean_proximity"]
housing_cat.head()

# each of values represent a category. most machine learing 
# algorithms prefer to work with numbers, so let's convert these
# categories from text to numbers. for this we use scikit-learn
# "OrdinalEncoder" class

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

# to get the list of categories:
# the list is a 1D array of categories for each categorical
# attribute(since there is just one categorical attribute)

ordinal_encoder.categories_

# one issue with this representation is that ML algorithms will 
# assume that two nearby values are more similar than two distant
# values. this may be fine in some cases( e.g for ordered categories)
# such as "bad", "average", "good",and "excellent". but it is
# obviously not the case for the "ocean_proximity" column
# Here we use "one-hot encoding" because only one attribute will
# be equal to 1(hot), while the others will be 0(cold).
#
# scikit-learn provides a "OneHotEncoder" class to convert categorical
# values into one-hot vectors

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot