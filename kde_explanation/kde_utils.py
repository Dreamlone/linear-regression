from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_kde_plots_path():
    models_folder = Path(get_project_path(), 'kde_explanation', 'kde_plots')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder


def get_kde_simulation_path():
    models_folder = Path(get_project_path(), 'kde_explanation', 'kde_simulation')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder
