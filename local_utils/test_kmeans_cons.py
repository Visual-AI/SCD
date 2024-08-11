from k_means_constrained.k_means_constrained_ import KMeansConstrained
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [4, 3], [4, 4], [4, 1]])

clf = KMeansConstrained(n_clusters=2, size_min=2, size_max=5, random_state=0)

print(clf.fit_predict(X))

print(clf.cluster_centers_, clf.labels_)

import torch
from sskm_constrained import K_Means as CSSKeans
clf = CSSKeans(k=2, size_min=2, size_max=5, random_state=0)
clf.fit(torch.from_numpy(X).float())
print(clf.cluster_centers_, clf.labels_)