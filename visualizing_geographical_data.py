# So far you have only taken a quick glance at the data to get a general
# understanding of the kind of data you are manipulating. Now the goal is to go
# into a little more depth.
# First, make sure you have put the test set aside and you are only exploring the
# training set. Also, if the training set is very large, you may want to sample an
# exploration set, to make manipulations easy and fast. In our case, the set is
# quite small, so you can just work directly on the full set. Let’s create a copy so
# that you can play with it without harming the training set:


housing = strat_train_set.copy()

# since there is geographical information(latitude and longitude)
# it is a good idea to craete a scatterplot of all districts to 
# visualize the data

housing.plot(kind="scatter", x="longitude" , y="latitude")

# it is more easier to visualize the places where there is a high
# density of data points by setting alpha option to 0.1

housing.plot(kind="scatter", x="longitude" , y="latitude", alpha=0.1)


# Now let's look at the housing prices. The radius of each circle
# represents the district's population(option s), and the color 
# represents the price( option c). we will use a predefined color
# map ( option cmap) called jet, which ranges from blue(low values)
# to red ( high prices)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100), label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()

# This image tells you that the housing prices are very much related to the
# location (e.g., close to the ocean) and to the population density, as you
# probably knew already. A clustering algorithm should be useful for detecting
# the main cluster and for adding new features that measure the proximity to the
# cluster centers. The ocean proximity attribute may be useful as well, although
# in Northern California the housing prices in coastal districts are not too high,
# so it is not a simple rule

#----------------------------------------------------------------
# looking for correlations between attributes in the data
# since the dataset is not too large, you can easily compute the 
# standard correlation coefficient( pearson's r ) between every
# pair of atrributes using the corr() method.

corr_matrix = housing.corr()

# Now let's look at how much each attribute correlates with the
# median house value

corr_matrix["median_house_value"].sort_values(ascending=false)


#----------------------------------------------------------------
# Another way to check for correlation between attributes is to 
# use the pandas "scatter_matrix()" function, which plots every
# numerical attribute. since there are now 11 numerical attributes,
# you would get 11 square = 121 plots, which would not fit on a 
# page. so we just focus on a few promising attributes that seem 
# most correlated with the median housing value

from pandas.plotting import scatter_matrix
attributes = ["median_house_value","median_income","total_rooms"
,"housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))

#---------------------------------------------------------------
# The most promising attribute to predict the median house value
# is the median income, so let's zoom in on their correlation
# scatterplot

housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)


# This plot reveals a few things. First, the correlation is indeed very strong; you
# can clearly see the upward trend, and the points are not too dispersed. Second,
# the price cap that we noticed earlier is clearly visible as a horizontal line at
# $500,000. But this plot reveals other less obvious straight lines: a horizontal
# line around $450,000, another around $350,000, perhaps one around $280,000,
# and a few more below that. You may want to try removing the corresponding
# districts to prevent your algorithms from learning to reproduce these data
# quirks.

