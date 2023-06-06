import random
import numpy as np
from scipy.optimize import minimize

import utils
import cluster

seed = 2023
np.random.seed(seed)  
random.seed(seed) 

def optimize(Y, y, iterator, sigma=1, alpha=2):
    def objective(c):
        return np.sum(1-np.exp(alpha * np.log(np.exp(-sigma*c)+np.exp(sigma*c)) / sigma))
    def constraint1(c, Y, y):
        return np.dot(Y, c) - y
    def constraint2(c, Y, iterator):
        return c[iterator]

    n, m = Y.shape
    x0 = np.zeros(m)
    bounds = [(None, None)] * m
    for i in range(m):
        bounds[i] = (0, None)
    constraints = [{'type': 'eq', 'fun': constraint1, 'args': (Y, y)},
                   {'type': 'eq', 'fun': constraint2, 'args': (Y, iterator)}]
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

if __name__ == '__main__':
    # signal load
    S = np.load("./data/S_BSS.npy")
    A = np.load("./data/A_BSS.npy")
 
    X = np.dot(A, S)
    X = X / np.max(X)

    # mixing matrix sloving
    spare_threshold = 1e-1      # L1: 1e-1  Our: 5
    cluster_k       = 3
    sigma           = 1        
    alpha           = 1         # L1: 10

    Omega = []
    for j in range(X.shape[1]):
        Y = X.copy()
        y = X[:, j]
        c = optimize(Y, y, iterator=j, sigma=sigma, alpha=alpha)

        c_abs = np.abs(c)
        threshold = spare_threshold
        c_abs[c_abs < threshold] = 0

        count = np.count_nonzero(c_abs)
        if count == 1:
            Omega.append(y)
        
        print(j, "-th sparse solving.", count, " non zero elements.")

    Omega = np.array(Omega)
    print(Omega.shape)
    Omega = Omega / np.max(Omega)
    
    labels, centroids = cluster.kmeans(data=Omega, k=cluster_k, max_iters=50000)
    centroids = centroids.T
    print("\nMixing Matrix:\n", A)
    print("\nEstimated Mixing Matrix:\n", centroids)

    if A.shape[0] == A.shape[1]:
        A_hat = np.linalg.inv(centroids)
        S_hat = np.dot(A_hat, X)
        utils.display(S, X, S_hat, signal_length=S.shape[1])