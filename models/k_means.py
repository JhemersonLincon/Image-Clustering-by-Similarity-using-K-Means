import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps 
            
class KMeans:
    def __init__(self, n_cluster=2, max_iter=10):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_cluster, replace=False)]
        for _ in range(0, self.max_iter):
            print(f"iter: {_}")
            self.labels = self.distancia_euclidiana(X=X)
            
            new_centroids = self.upgrade_centroids(X, labels=self.labels)
            
            if np.allclose(new_centroids, self.centroids, atol=1e-4):
                print(f"Convergiu na iteração: {_}")
                break


            self.centroids = new_centroids
            
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def distancia_euclidiana(self, X):
        distance = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distance, axis=1)
    
    def upgrade_centroids(self, X, labels):
        new_centroids = [X[labels==i].mean(axis=0) for i in range(0, self.n_cluster)]
        return np.array(new_centroids, dtype=np.float32)