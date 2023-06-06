import numpy as np
from sklearn.cluster import KMeans

def kmeans(data, k, max_iters=5000):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for i in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        for j in range(k):
            centroids[j] = data[labels == j].mean(axis=0)
    return labels, centroids

def kmeans_pp(data, k, max_iters=5000):
    centroids = [data[np.random.choice(len(data))]]
    for i in range(k-1):
        distances = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                i = j
                break
        centroids.append(data[i])
    kmeans = KMeans(n_clusters=k, init=np.array(centroids), max_iter=max_iters)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids 