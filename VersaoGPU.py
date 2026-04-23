import numpy as np
import cupy as cp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix


def wasserstein_1d(xi, xj):
    xi = cp.sort(xi)
    xj = cp.sort(xj)
    return cp.mean(cp.abs(xi - xj))


def gtsa_pca_faithful(X, k, p, K, tau=1.0, mode="curvature"):
    n, D = X.shape

    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    _, idx = knn.kneighbors(X)

    Xg = cp.asarray(X)
    Kg = cp.asarray(K)

    U = cp.zeros((n, D, p))

    for i in range(n):
        neigh = idx[i, 1:]
        Xi = Xg[i]
        Xj = Xg[neigh]

        W = cp.zeros(k)

        for t, j in enumerate(neigh):
            if mode == "curvature":
                W[t] = cp.exp(-cp.abs(Kg[j]) / tau)
            else:
                W[t] = wasserstein_1d(Xg[i], Xg[j])

        Zi = cp.sum(W) + 1e-12
        W = W / Zi

        diff = Xj - Xi
        Sigma = cp.zeros((D, D))

        for t in range(k):
            v = diff[t][:, None]
            Sigma += W[t] * (v @ v.T)

        eigvals, eigvecs = cp.linalg.eigh(Sigma)
        order = cp.argsort(eigvals)[::-1]
        U[i] = eigvecs[:, order[:p]]

    U_cpu = cp.asnumpy(U)

    rows, cols, vals = [], [], []

    for i in range(n):
        for t, j in enumerate(idx[i, 1:]):
            rows.append(i)
            cols.append(j)
            vals.append(np.linalg.norm(X[i] - X[j]))

    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    dG = dijkstra(G, directed=False)

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sij = np.trace(U_cpu[i].T @ U_cpu[j])
            A[i, j] = (1.0 / (1.0 + dG[i, j])) * sij if not np.isinf(dG[i, j]) else 0.0

    eigvals, eigvecs = np.linalg.eigh(A)
    Y = eigvecs[:, np.argsort(eigvals)[::-1][:p]]

    return Y
