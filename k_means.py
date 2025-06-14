import numpy as np

class KMeans:
    def __init__(self, n_cluster=2, max_iter=10):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        
    def fit(self, X):
        X = X.reshape(X.shape[0], 1)
        self.centroids = X[np.random.choice(X.shape[0], self.n_cluster, replace=False)]
        self.centroids = self.centroids.reshape(1, self.centroids.shape[0])
        for _ in range(0, self.max_iter):
            self.labels = self.distancia_euclidiana(X=X)
            
            new_centroids = self.upgrade_centroids(X, labels=self.labels)
            
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids
            
    
    def distancia_euclidiana(self, X):
        distance =    np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=1)
        return np.argmin(distance, axis=1)
    
    def upgrade_centroids(self, X, labels):
        new_centroids = [X[labels==i].mean(axis=0) for i in range(0, self.n_cluster)]
        return np.array(new_centroids, dtype=np.float32).reshape(1, self.n_cluster)
