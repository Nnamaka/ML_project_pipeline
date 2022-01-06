
# you identified a few data quirks that you may want to clean up
# before feeding the data to a machine learning algorithm, and
# you found interesting correlations between attributes, in 
# particular with the target attribute.
#------------------------------------------------------------
# one last thing you may want to do before preparing the data 
# for machine learning algorithms is to try out various attribute
# combinations.
# For example,
# the total number of rooms in a district is not very useful if you don’t know
# how many households there are. What you really want is the number of rooms
# per household. Similarly, the total number of bedrooms by itself is not very
# useful: you probably want to compare it to the number of rooms. And the
# population per household also seems like an interesting attribute combination
# to look at. Let’s create these new attributes:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# and now let's look at the correlation matrix again
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Hey, not bad! The new bedrooms_per_room attribute is much more correlated
# with the median house value than the total number of rooms or bedrooms.
# Apparently houses with a lower bedroom/room ratio tend to be more
# expensive. The number of rooms per household is also more informative than
# the total number of rooms in a district—obviously the larger the houses, the
# more expensive they are.
# This round of exploration does not have to be absolutely thorough; the point is
# to start off on the right foot and quickly gain insights that will help you get a
# first reasonably good prototype.



