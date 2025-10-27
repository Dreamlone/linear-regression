from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_plots_path():
    models_folder = Path(get_project_path(), 'plots')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder


def get_plots_templates_path():
    models_folder = Path(get_project_path(), 'plots_templates')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder


def get_tmp_animation_directory():
    plots_folder = Path(get_plots_path(), 'animation')
    plots_folder.mkdir(parents=True, exist_ok=True)
    return plots_folder


def get_results_path():
    models_folder = Path(get_project_path(), 'results')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder
