# K-Means Clustering

This is the implementation of K-Means Clustering algorithm which is an unsupervised learning model. It aims to separate the data points into k clusters in which each data point is assigned to the cluster with the closest centroid.
The data points which are in the same cluster have similar features.

### How does the algorithm work?

1) Select the number of clusters (K).
2) Randomly select K initial centroids.
3) Repeat the process below until the centroids don't change: 
   * Form K clusters assigning each data point to the closest centroid (Measure the distance of data points to all initial centroids, assign the data point to the nearest cluster).
   * Recompute the centroid of each cluster by taking the mean of all the vectors.

### Output

When implementing the algorithm, I have created a dataset of size 500 with four center points. Since there are four centroids, the model separates the data points into four clusters.


![image](https://user-images.githubusercontent.com/61224886/96360009-9537ff00-1121-11eb-8b2e-558c66918e87.png)
