
# 1. This will allow you to reprodduce these transformations easily
# on any dataset(e.g, the next time you get a fresh dataset)
# 2. You will gradually build a library of transformation functions
# That you can reuse in future projects.
# 3. You can use these functions in your live system to transform
# the new data before feeding it to your algorithms.
# 4. This will make it possible for you to easily try various
# transformations and see which combinaton of transformations works
# best.

# but first let's revert to a clean training set(by copying "
# strat_train_set" once aganin).

housing - strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
