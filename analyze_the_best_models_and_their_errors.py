
# You will often gain good insights on the problem by inspecting the best
# models. For example, the RandomForestRegressor can indicate the relative
# importance of each attribute for making accurate predictions:


feature_importances = grid_search.best_estimator_.feature_importance_
feature_importances

# let's display these importance scores next to their corresponding
# attribute names:

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])

attibutes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes) , reverse=True)

# with this information, you may want to try dropping some ot the 
# less useful features(e.g, apparently only one "ocean_proximity"
# # category is really useful, so you could try dropping the others)

# you should also look at the specific errors that your system makes
# then try to understand why it makes them and what could fix the 
# problem (adding extra features or getting rid of uninformative ones,
# cleaning up outliers, etc)

