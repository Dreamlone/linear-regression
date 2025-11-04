import numpy as np

# Given data
rooms = np.array([1, 2, 4])
prices = np.array([10000, 20000, 40000])
areas = np.array([30, 50, 90])  # New feature (area in square meters)

# Convert data to matrix form
X = np.column_stack((rooms, areas))
y = prices

# Adding bias term (column of ones)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Compute coefficients using normal equation (which won't work due to potential singularity)
try:
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    print(f"Estimated coefficients: {B}")
except np.linalg.LinAlgError:
    print("Analytical solution cannot be computed due to singularity or ill-conditioned matrix.")

"""
Visualize this
B = np.linalg.inv(X.T @ X) @ X.T @ prices 

in a form of matrices and vectors, for example here we know that prices is a vector [10000, 20000, 40000], rooms = np.array([1, 2, 4]) and areas = np.array([30, 50, 90]). And X.T means transpose matrix X which is X = np.hstack((np.ones((X.shape[0], 1)), X)). So show me matrices and vectors with numbers
"""
