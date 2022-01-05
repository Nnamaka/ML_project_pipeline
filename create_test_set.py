# the code below automates creating a test set by defining a 
# function
np.random.seed(42)

def split_train_test( data , test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# calling the functioin
# note that "housing" is a panda data frame
train_set, test_set = split_train_test( housing, 0.2)

#-------------------------------------------------
# if you run the above program again, it will generate a 
# different test set!
# over time, you(or your Machine Learning algorithms) will
# get to see the whole dataset. ie it keeps bringing/generating
# different values from the whole dataset until the ML algo
# sees all the datasets. Two ways to solve this is to 
# 1. Save the test set on the first run and then load it in 
# subsequent runs.
# 2. Set the random number generator's seed (np.random.seed(42))
# before calling (np.random.permutation()) so that it always
# generates the same shuffled indices as implemented in the code
# above.

#---------------------------------------------------
# But both these solutions will break the next time you fetch an
# updated dataset.
# note: lookup "what is checksum" in google. checksum is similar
# to hashing.
# a common
# solution is to use each instance’s identifier to decide whether or not it should
# go in the test set (assuming instances have a unique and immutable
# identifier). For example, you could compute a hash of each instance’s
# identifier and put that instance in the test set if the hash is lower than or equal
# to 20% of the maximum hash value. This ensures that the test set will remain
# consistent across multiple runs, even if you refresh the dataset. The new test
# set will contain 20% of the new instances, but it will not contain any instance
# that was previously in the training set.


from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    # note data is a pandas dataframe
    # id_column is an identifier column for the checksum/hashing
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Unfortunately, the housing dataset does not have an identifier column. The
# simplest solution is to use the row index as the ID:
# "housing" is a pandas dataframe

housing_with_id = housing.reset_reset_index() # adds an 'index' column
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")

# If you use the row index as a unique identifier, you need to make sure that
# new data gets appended to the end of the dataset and that no row ever gets
# deleted. If this is not possible, then you can try to use the most stable features
# to build a unique identifier. For example, a district’s latitude and longitude are
# guaranteed to be stable for a few million years, so you could combine them
# into an ID like so:

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2,"id")


# Scikit-Learn provides a few functions to split datasets into multiple subsets in
# various ways. The simplest function is train_test_split(), which does
# pretty much the same thing as the function split_train_test(), with a
# couple of additional features. First, there is a random_state parameter that
# allows you to set the random generator seed. Second, you can pass it multiple
# datasets with an identical number of rows, and it will split them on the same
# indices (this is very useful, for example, if you have a separate DataFrame for
# labels):

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)







