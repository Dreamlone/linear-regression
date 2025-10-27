import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, verbose=4)
target = np.array([[0],
                   [0],
                   [1],
                   [1],
                   [0],
                   [0],
                   [0],
                   [1],
                   [1],
                   [1]])
clf.fit(np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4],
                  [5, 5, 5],
                  [6, 6, 6],
                  [7, 7, 7],
                  [8, 8, 8],
                  [9, 9, 9],
                  [10, 10, 10]]), np.ravel(target))
print(sorted(clf.cv_results_.keys()))