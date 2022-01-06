
# Although Scikit-Learn provides many useful transformers, you will need to
# write your own for tasks such as custom cleanup operations or combining
# specific attributes. You will want your transformer to work seamlessly with
# Scikit-Learn functionalities (such as pipelines), and since Scikit-Learn relies
# on duck typing (not inheritance), all you need to do is create a class and
# implement three methods: fit() (returning self), transform(), and
# fit_transform().
#
# you can get the last one free by simply adding "TransformerMixin"
# as a base class. if you add "BaseEstimator" as a base class
# (and avoid *args and **kargs in your constructor), you will also
# get two extra methods (get_params() and set_params()) that will
# be useful for automatic hyperparameter tunning

# below is a small transformer class that adds the combined 
# attributes we discussed earlier

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# in this example the transformer has one hyperparameter,
# add_bedrooms_per_room, set to True by default (it is often helpful to provide
# sensible defaults). This hyperparameter will allow you to easily find out
# whether adding this attribute helps the Machine Learning algorithms or not.
# More generally, you can add a hyperparameter to gate any data preparation
# step that you are not 100% sure about. The more you automate these data
# preparation steps, the more combinations you can automatically try out,
# making it much more likely that you will find a great combination (and
# saving you a lot of time).




