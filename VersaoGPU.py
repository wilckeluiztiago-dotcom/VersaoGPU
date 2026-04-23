import cupy as cp
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

def gtsa_pca_gpu(X, k, p, curv, tau=1.0, modo="curvatura"):
    n, D = X.shape

    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    _, idx = knn.kneighbors(X)

    Xg = cp.asarray(X)
    curv_g = cp.asarray(curv)

    tangentes = cp.zeros((n, D, p))

    for i in range(n):
        neigh_idx = idx[i, 1:]
        Xi = Xg[i]
        V = Xg[neigh_idx]

        diff = V - Xi

        if modo == "curvatura":
            w = cp.exp(-cp.abs(curv_g[neigh_idx]) / tau)
        else:
            w = cp.linalg.norm(diff, axis=1)

        w = w / (cp.sum(w) + 1e-9)

        cov = cp.zeros((D, D))
        for j in range(k):
            v = diff[j][:, None]
            cov += w[j] * (v @ v.T)

        eigvals, eigvecs = cp.linalg.eigh(cov)
        order = cp.argsort(eigvals)[::-1]
        tangentes[i] = eigvecs[:, order[:p]]

    tangentes_cpu = cp.asnumpy(tangentes)

    rows, cols, vals = [], [], []

    for i in range(n):
        for j_idx, j in enumerate(idx[i, 1:]):
            rows.append(i)
            cols.append(j)
            vals.append(np.linalg.norm(X[i] - X[j]))

    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    geo = dijkstra(G, directed=False)

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            align = np.trace(tangentes_cpu[i].T @ tangentes_cpu[j])
            d = geo[i, j]
            if np.isinf(d):
                d = 1e9
            A[i, j] = align / (1.0 + d)

    eigvals, eigvecs = np.linalg.eigh(A)
    Y = eigvecs[:, np.argsort(eigvals)[::-1][:p]]

    return Y