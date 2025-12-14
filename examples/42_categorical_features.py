from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder

from examples.paths import get_plots_path
from examples.utils import get_extended_dataset

FONTNAME = "Comic Sans MS"

def encode_examples():
    dataset = get_extended_dataset()
    # Prune dataset so it will be easy for visualization
    df = dataset.sample(n=6, random_state=21).sort_values(by=["city", "rooms"])

    print("LabelEncoder")
    categorical_features = np.array(df[["city", "ac_in_apartment"]])
    for i in [0, 1]:
        enc = LabelEncoder()
        encoded = enc.fit_transform(categorical_features[:, i])
        print(i, encoded)

    print("OneHotEncoder")
    categorical_features = np.array(df[["city", "ac_in_apartment"]])
    enc = OneHotEncoder()
    encoded = enc.fit_transform(categorical_features).toarray()
    print(encoded)

    print("TargetEncoder (self-made)")
    for i in ['city', 'ac_in_apartment']:
        mean_target_by_category = (
            df
            .groupby(i)['price']
            .mean()
        )
        print(mean_target_by_category)


if __name__ == '__main__':
    encode_examples()
