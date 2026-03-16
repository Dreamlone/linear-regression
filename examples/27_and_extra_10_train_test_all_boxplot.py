from pathlib import Path
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

import matplotlib.pyplot as plt
import pandas as pd

from examples.paths import get_plots_path, get_results_path
from examples.utils import save_plot_according_to_template

FONTNAME = "Comic Sans MS"
FONTDICT = {'fontsize': 14, 'fontname': FONTNAME}
BEST_EXPECTED_RMSE_ALL = 3873.270


def annotations_by_language(mode: str):
    if mode == "eng":
        sample_label = "Sample"
        experiment_setup_label = "Sample size used\nto fit the model"
        title = "RMSE values across different samples"
        train = "train"
        test = "test"
        all = "full population"
        text_label = f"RMSE of the model\nfitted on the\nfull population\n{BEST_EXPECTED_RMSE_ALL:.1f}\nest. entire data"
    elif mode == "rus":
        sample_label = "Выборка"
        experiment_setup_label = "Размер семпла по\nкоторому строилась модель"
        title = "Значения метрики RMSE на разных выборках"
        train = "обучение"
        test = "тест"
        all = "генеральная совокупность"
        text_label = f"RMSE модели,\nпостроенной на\nген. совокупности\n{BEST_EXPECTED_RMSE_ALL:.1f}"
    else:
        raise NotImplementedError(f"Language {mode} is not supported")
    return sample_label, experiment_setup_label, title, train, test, all, text_label


def read_data_as_long_format(mode):
    sample_label, experiment_setup_label, title, train, test, all, text_label = annotations_by_language(mode)

    df = pd.read_csv(Path(get_results_path(), "train_test_all_metrics.csv"))
    df_train = df[["experiment_setup", "id", "rmse train"]].rename(columns={"rmse train": "RMSE",
                                                                            "experiment_setup": experiment_setup_label})
    df_train[sample_label] = train

    df_test = df[["experiment_setup", "id", "rmse test"]].rename(columns={"rmse test": "RMSE",
                                                                          "experiment_setup": experiment_setup_label})
    df_test[sample_label] = test

    df_all = df[["experiment_setup", "id", "rmse all"]].rename(columns={"rmse all": "RMSE",
                                                                        "experiment_setup": experiment_setup_label})
    df_all[sample_label] = all
    df_long = pd.concat([df_train, df_test, df_all])

    df_long[experiment_setup_label] = df_long[experiment_setup_label].replace({"small": 10, "big": 20})
    return df_long


def plot_all_datasets_in_one(mode: str = "eng"):
    """
    Generate the plot with linear regression line and equation

    To generate english plot choose mode "eng"
    To generate russian - "rus"
    """
    sample_label, experiment_setup_label, title, train, test, all, text_label = annotations_by_language(mode)

    # Read the data from previous step
    df: pd.DataFrame = read_data_as_long_format(mode)

    # Show some statistics
    aggregated = df.groupby([experiment_setup_label, sample_label]).agg({"RMSE": "mean"})
    print(aggregated)

    #############
    # STRIPPLOT #
    #############
    fig_size = (9, 6)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = sns.stripplot(x=sample_label, y="RMSE", hue=experiment_setup_label,
                       palette=["r", "b"], data=df, ax=ax, dodge=True)
    legend = plt.legend(loc='upper left', title=experiment_setup_label)
    legend.get_title().set_ha('center')
    legend.get_title().set_fontname(FONTNAME)
    for text in legend.get_texts():
        text.set_fontname(FONTNAME)

    ax.plot([-0.25, 0.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=0.5)
    ax.plot([0.75, 1.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=0.5)

    ax.plot([1.75, 2.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=1.0)

    for x_coord in [2.5]:
        ax.text(x_coord, BEST_EXPECTED_RMSE_ALL, text_label, fontname=FONTNAME,
                fontsize=8, c='orange', va='center', ha='center')
    ax.set_xlabel(sample_label, fontname=FONTNAME, fontsize=14)
    ax.set_ylabel("RMSE", fontname=FONTNAME, fontsize=14)

    sns.despine(offset=10, trim=True)
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(get_plots_path(), f"27_train_test_all_stripplot_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"27_train_test_all_stripplot_{mode}.png"))

    ###########
    # BOXPLOT #
    ###########
    fig_size = (9, 6)
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax = sns.boxplot(x=sample_label, y="RMSE", hue=experiment_setup_label,
                     palette=["r", "b"], data=df, ax=ax, width=0.5)
    legend = plt.legend(loc='upper left', title=experiment_setup_label)
    legend.get_title().set_ha('center')
    legend.get_title().set_fontname(FONTNAME)
    for text in legend.get_texts():
        text.set_fontname(FONTNAME)

    ax.plot([-0.25, 0.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=0.5)
    ax.plot([0.75, 1.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=0.5)

    ax.plot([1.75, 2.25], [BEST_EXPECTED_RMSE_ALL, BEST_EXPECTED_RMSE_ALL], '--', c='orange', linewidth=2, alpha=1.0)

    for x_coord in [2.5]:
        ax.text(x_coord, BEST_EXPECTED_RMSE_ALL, text_label, fontname=FONTNAME,
                fontsize=8, c='orange', va='center', ha='center')
    ax.set_xlabel(sample_label, fontname=FONTNAME, fontsize=14)
    ax.set_ylabel("RMSE", fontname=FONTNAME, fontsize=14)

    sns.despine(offset=10, trim=True)
    fig.suptitle(title, fontsize=16, fontdict={'fontname': FONTNAME})

    raw_svg_file = Path(get_plots_path(), f"extra_10_train_test_all_boxplot_{mode}.svg")
    plt.savefig(raw_svg_file)
    plt.close()

    save_plot_according_to_template(raw_svg_file,
                                    Path(get_plots_path(), f"extra_10_train_test_all_boxplot_{mode}.png"))


if __name__ == '__main__':
    plot_all_datasets_in_one("rus")
    plot_all_datasets_in_one("eng")
