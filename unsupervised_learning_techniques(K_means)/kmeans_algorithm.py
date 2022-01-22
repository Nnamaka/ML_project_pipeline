
# clustering is the task of identifying similar instances and 
# assigning them to clusters, or groups of similar instances.
# just like in classification, each instances gets assigned to a
# group. However, unlike classification, clustering is an 
# unsupervised task.

# Uses of Clustering algorithms(kmeans)
# 1. customer segmentation
# 2. For data analysis
# 3. As a dimensionality reduction technique
# 4. For anomaly detection
# 5. For semi-supervised learning
# 6. search Engines
# 7. To segment an image(image segmentation)

# The Kmeans algorithm is guaranteed to converge, it may not 
# converge to the right solution(i.e it may converge to a local
# optimum): whether it does or not depends on the centroid initializtion

# The number of random initializations is controlled by the n_init 
# hyperparameter: by default, it is equal to 10, which means that 
# the whole algorithm described earlier runs
# 10 times when you call fit(), and Scikit-Learn keeps the best solution.
# But how exactly does it know which solution is the best? It uses a
# performance metric! That metric is called the model’s inertia, which is the
# mean squared distance between each instance and its closest centroid

# "K-Means++" is has a smarter initialization step that tends to 
# select centroids that are distant from one another, and this 
# improvement makes the K-means algorithm much less likely to 
# converge to a suboptimal solution.

#---------------------------------------------------------------
# HOW TO CHOOSE THE BEST VALUE FOR THE NUMBER OF CLUSTERS
# 1. Elbow
# 2. silhouette score( more precise but computationally expensive
# approach). To compute the silhouette score in Scikit-learn see
# below:

from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)