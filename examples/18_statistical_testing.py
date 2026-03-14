from pathlib import Path

import statsmodels.api as sm
from scipy.stats import f

import numpy as np
import matplotlib.pyplot as plt

import warnings

from examples.paths import get_plots_path

warnings.filterwarnings('ignore')


def build_model_and_test(significance_level: float = 0.05):
    """
    This is very simple script which builds 1 plot
    where visualizes the F statistics and p-value
    """
    y = np.array([3000, 29000, 21000])
    features = np.array([1, 2, 3])

    feature_in_model = sm.add_constant(features)

    # Build the model
    model = sm.OLS(y, feature_in_model).fit()
    predicted = model.predict(feature_in_model)

    print(f"Predicted values: {predicted}")
    print(model.summary())

    # Calculate critical F value "manually"
    df_model = model.df_model
    df_resid = model.df_resid
    f_critical = f.ppf(1 - significance_level, df_model, df_resid)

    print(f"F-statistic: {model.fvalue:.2f}")
    print(f"Critical F-value (alpha={significance_level}): {f_critical:.2f}")
    print(f"F-statistic p-value: {model.f_pvalue:.2f}")

    fig_size = (5, 11)
    fig, axs = plt.subplots(3, 1, figsize=fig_size)

    axs[0].bar(["F statistic", "Critical F-value"], [model.fvalue, f_critical], color='orange', width=0.2)
    axs[1].bar(["p-value", "significance level"], [model.f_pvalue, significance_level], color='orange', width=0.2)
    axs[2].scatter(features, y, c='black')
    axs[2].plot(features, predicted, '--', c='black')
    axs[2].plot(features, [np.mean(y), np.mean(y), np.mean(y)], '--', c='black')

    raw_svg_file = Path(get_plots_path(), f"18_model_testing_raw.svg")
    plt.savefig(raw_svg_file)
    plt.close()


if __name__ == '__main__':
    build_model_and_test()
