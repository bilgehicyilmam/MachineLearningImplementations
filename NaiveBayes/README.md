**GAUSSIAN NAIVE BAYES**
	
This is the implementation of the Gaussian Naive Bayes algorithm. In GNB, we assume that continuous values are distributed 
according to normal (Gaussian) distribution. 
	
**How does the Gaussian Naive Bayes algorithm work?**
	
1. Separate dataset by their classes and calculate the mean, standard deviation, and prior probabilities of each class.
2. For **each data point** in the dataset, calculate the conditional probability **for each class** with Normal density function, and multiply conditional probabilities with prior probabilities to get posterior probabilities.
3. Choose the class with the highest probability value.
