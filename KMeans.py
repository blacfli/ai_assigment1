# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:19:05 2021

@ Author: Michael Evan Santoso
@ Student Number: 26001904409
@ Date: 20 December 2021
    Program for K-Means clustering algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm as euclidean

class KMeans:
    def __init__(self, n_clusters, *, method = 'k-means++', n_init = 20, max_iteration = 200, random_state = None, verbose = False):
        """
        Initialize all the parameter and variable that will be used for K-Means.

        Parameters
        ----------
        n_clusters : int
            Number of clusters which is decided by user.
        method : str, optional
            Method that is used for initializing the centroids. The default is 'k-means++'.
        n_init : int, optional
            Number of different attemp to find the minimum inertia. The default is 20.
        max_iteration : int, optional
            Number of iteration for making sure K-means converge. The default is 200.
        random_state : None or int, optional
            Random seed for it to become reproducible. The default is None.
        verbose : bool, optional
            To define the verbosity mode. The default is False.

        Returns
        -------
        None.

        """
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        self.max_iteration = max_iteration
        self.n_init = n_init
        self.verbose = verbose
        
        # To initialize the global inertia, best centroids, labels, and subset
        self.inertia_ = 0 
        self.best_centroids_ = None
        self.labels_ = None
        self.subset_ = {}
    
    def _init_centroids(self, X):
        """
        Initialize the centroids depending the method that is used

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.

        Returns
        -------
        centroids : numpy array
            The initialized centroids for the K-means algorithm.

        """
        
        # Setting condition if the user want to use Initial K-Means method or
        # using K-Means++ method.
        if(self.method == 'k-means++'):
            centroids = X[np.random.choice(X.shape[0], size = self.n_clusters, replace = False)]
        elif(self.method == 'k-means'):
            centroids = np.random.uniform(np.min(X), np.max(X), size = (self.n_clusters, X.shape[1]))
        return centroids
    
    def _calculate_inertia(self, X, centroids, labels):
        """
        Function to calculate the inertia for the given centroids

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.
        centroids : numpy array
            The current centroids of the K-means algorithm.
        labels : numpy array
            The current labels of the K-means algorithm.

        Returns
        -------
        inertia : float
            The resulting inertia of the given centroids.

        """
        
        # Set the inertia to 0
        inertia = 0
        
        # Define the unique label
        unique_labels = np.unique(labels)
        
        # FOR loop for calculating the inertia
        for unique_label in unique_labels:
            temp_X = X[np.where(labels == unique_label)]
            squared_norm = temp_X - centroids[unique_label]
            inertia += euclidean(squared_norm) ** 2
        return inertia
        
    def fit(self, X):
        """
        Train the K-means model to find the clusters

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.

        Returns
        -------
        None.

        """
        
        # Set the seed to that the result will be deterministic
        np.random.seed(self.random_state)
        
        # Define list of labels, centroids, and inertia to 
        # find the best result
        labels = []
        centroids = []
        inertia = []
        
        # Iterating for finding the best inertia
        for init in range(self.n_init):
            # Using verbose to print out the result
            if(self.verbose and (init + 1) % 10 == 0):
                print('Initilization', init + 1)
                
            # Initialize the centroids, current center, temporary center,
            # and temporary variable.
            initial_centroids = self._init_centroids(X)
            current_center = initial_centroids
            temp_center = current_center.copy()
            temp = np.zeros(shape = (self.n_clusters, X.shape[0]))
            
            # FOR loop for defining the maximum iteration that 
            # the algorithm runs
            for iteration in range(self.max_iteration):
                # Using verbose to print out the result
                if self.verbose:
                    print('Iteration', iteration + 1)
                # FOR loop for calculating each euclidean distance of data
                # from the centroid
                for idx, center in enumerate(current_center):
                    temp[idx] = euclidean(X - center, axis = 1)
                
                # Selecting the subset from the centroids and label them
                select_centroids = np.argmin(temp, axis = 0)
                
                # Define unique centroids for the next step
                unique_centroids = np.unique(select_centroids)
                
                # FOR loop for calculating the mean of each new centroids
                for unique_centroid in unique_centroids:
                    current_center[unique_centroid] = np.mean(X[np.where(select_centroids == unique_centroid)], axis = 0)
                
                # Stop if there is no change for the current centroids and 
                # the previous centroids
                if euclidean(current_center - temp_center) == 0:
                    # Using verbose to print out the result
                    if self.verbose:
                        print('Converged in iteration', iteration + 1)
                    break
                
                # Set the temporary centroid after checking the converging
                # condition
                temp_center = current_center
            
            # Append labels, centroids, and inertia, to be selected later
            # for the best value in the inertia
            calculated_inertia = self._calculate_inertia(X, current_center, select_centroids)
            labels.append(select_centroids)
            centroids.append(current_center)
            inertia.append(calculated_inertia)
        
        # Find the index of the minimum inertia
        idx_best = np.argmin(inertia)
        
        # Put the final labels, best centroids, and inertia to
        # global value
        self.labels_ =  labels[idx_best]
        self.best_centroids_ = centroids[idx_best]
        self.inertia_ = self._calculate_inertia(X, self.best_centroids_, self.labels_)
        
        # To make the subset for each clusters
        for label in self.labels_:
            self.subset_[label] = list(np.where(self.labels_ == label)[0] + 1)
        
    def predict(self, X):
        """
        Function to predict the labels that the feature or data points
        belong to.

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.

        Returns
        -------
        select_centroids : numppy array
            Labels for the given clusters per data points.

        """
        temp = np.zeros(shape = (self.n_clusters, X.shape[0]))
        for idx, center in enumerate(self.best_centroids):
            temp[idx] = euclidean(X - center, axis = 1)
        select_centroids = np.argmin(temp, axis = 0)
        return select_centroids
    
    def fit_predict(self, X):
        """
        Function to train the K-means model to find cluster
        and predict the feature itself for the labels

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.

        Returns
        -------
        numpy array
            Labels for the given clusters per data points.

        """
        self.fit(X)
        return self.labels_
        
    def plot_graph(self, X):
        """
        Function to plot the graph of the K-Means clustering

        Parameters
        ----------
        X : numpy array
            The feature or variable to find the cluster.

        Returns
        -------
        None.

        """
        color = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        unique_labels = np.unique(self.labels_)
        
        plt.figure(figsize = [8, 8])
        plt.scatter(self.best_centroids_[:, 0], self.best_centroids_[:, 1], c = 'red', marker = '*', s = 200, label = 'Cluster Centroids')
        for idx, unique_label in enumerate(unique_labels):
            temp_X = X[np.where(self.labels_ == unique_label)]
            plt.scatter(temp_X[:, 0], temp_X[:, 1], c= color[idx], label = 'Cluster ' + str(unique_label))
        
        plt.title('K-Means Graph', fontweight = 'bold', fontsize = 20)
        plt.xlabel('Variable 1', fontsize = 18)
        plt.ylabel('Variable 2', fontsize = 18)
        plt.legend(loc = 'best')
        plt.show()
    
if __name__ == '__main__':
    
    # Define the feature to be clustered by K-Means
    feature = np.array([[1. , 1.],
                        [1.5, 2.],
                        [3., 4.],
                        [5., 7.],
                        [3.5, 5.],
                        [4.5, 5.],
                        [3.5, 4.5]])
    
    # Call the class for K-Means with 3 clusters, k-means++ method,
    # 100 maximum iteration, False verbosity
    km = KMeans(n_clusters = 3, method = 'k-means++', max_iteration = 100, verbose = False, random_state=2020)
    
    # Fit and find the cluster model with K-Means
    km.fit(feature)
    
    # Print the best centroids, labels, inertia, and subset
    print('Centroids: ')
    print(km.best_centroids_)
    print('labels: ', km.labels_)
    print('Inertia:', km.inertia_)
    print('Subset:', km.subset_)
    
    # Plot the K-Means the graph
    km.plot_graph(feature)
    