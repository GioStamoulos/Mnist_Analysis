# Mnist_Analysis
Script is related to 5 tasks.


**Task1** : Preprocessing ---> keep 2, 4, 6, 8 labels of dataset and convert to appropriate form.    
**Task2** : Convert every sample (image 28x28) to 2D array based on mean brightness of even rows (1st component) and mean brightness of odd columns (2nd component). 
**Task3** : Apply K-mean algorithm (from scratch) using Maximin algorithm for initialization of center of clusters (from scratch)  to previous 2D array and evaluate clustering through purity metric (supervised) with existing labels.
**Task4** : Apply dimensionality reduction with PCA (from scratch) to whole Mnist dataset (2, 4, 6 & 8 labels) and then apply K-mean algorithm (from task 3). 
**Task5** : Utilize Gaussian Naive Bayes Classifier (from scratch) for classification of dimensional reduced Mnist dataset (task4).
