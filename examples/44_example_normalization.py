from pathlib import Path
from typing import Dict, Union, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def show_normalization():
    np.set_printoptions(precision=2, suppress=True)

    features = np.array([[1, 10],
                         [2, 20],
                         [3, 505]])
    min_max = MinMaxScaler()
    transformed = min_max.fit_transform(np.copy(features))
    standard_scaler = StandardScaler()
    transformed_sc = standard_scaler.fit_transform(np.copy(features))

    print("Training sample. MinMaxScaler")
    print(transformed.round(2))
    print("Training sample. StandardScaler")
    print(transformed_sc.round(2))

    new_features = np.array([[0, 5],
                             [2, 20],
                             [100, 30]])
    print("New data. MinMaxScaler")
    new_transformed = min_max.transform(np.copy(new_features))
    print(new_transformed.round(2))

    print("New data. StandardScaler")
    new_transformed = standard_scaler.transform(np.copy(new_features))
    print(new_transformed.round(2))


if __name__ == "__main__":
    show_normalization()
