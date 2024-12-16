import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from aquarel import load_theme
from tqdm import tqdm
import pretty_errors  # Improves error traceback formatting
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="aquarel")



def crosstabs(data, columns: list, shape: tuple = (2, 2), figsize: tuple = (10, 10), params: dict = {}):
    """
    Generates heatmaps for crosstabs of column pairs.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to generate crosstabs for.
        shape (tuple): Grid shape for subplots (default: (2, 2)).
        figsize (tuple): Figure size for the plots (default: (10, 10)).
        params (dict): Additional parameters for sns.heatmap.
    """
    combs = list(itertools.combinations(columns, 2))
    assert len(combs) <= shape[0] * shape[1], "Grid shape is too small for the number of combinations."
    
    with load_theme("minimal_light"):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        for i, comb in enumerate(tqdm(combs, desc='Plotting crosstabs')):
            sns.heatmap(pd.crosstab(data[comb[0]], data[comb[1]]), ax=axs[i], annot=True, fmt='.0f', **params)
        plt.tight_layout()
        plt.show()

# Function 2: Bar Plots
def barplots(data, columns: list, shape: tuple = (2, 2), figsize: tuple = (10, 10), params: dict = {}):
    """
    Generates bar plots for value counts of specified columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to generate bar plots for.
        shape (tuple): Grid shape for subplots (default: (2, 2)).
        figsize (tuple): Figure size for the plots (default: (10, 10)).
        params (dict): Additional parameters for sns.barplot.
    """
    assert len(columns) <= shape[0] * shape[1], "Grid shape is too small for the number of columns."
    
    with load_theme("minimal_light"):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        for i, col in enumerate(tqdm(columns, desc='Plotting barplots')):
            temp = data[col].value_counts()
            sns.barplot(x=temp.index, y=temp.values, ax=axs[i], **params)
            axs[i].set_title(col)
        plt.tight_layout()
        plt.show()

# Function 3: Histograms
def histograms(data, shape: tuple, y: str = None, figsize: tuple = (20, 10), params: dict = {}):
    """
    Generates histograms for numerical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        shape (tuple): Grid shape for subplots.
        y (str): Optional column for grouping (default: None).
        figsize (tuple): Figure size for the plots (default: (20, 10)).
        params (dict): Additional parameters for sns.histplot.
    """
    x = [col for col in data.columns if col != y]
    assert len(x) <= shape[0] * shape[1], "Grid shape is too small for the number of columns."
    
    with load_theme("minimal_light"):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        
        if y:
            for i, col in enumerate(tqdm(x, desc=f'Plotting histograms grouped by {y}')):
                for cat in data[y].unique():
                    sns.histplot(data[data[y] == cat][col], label=str(cat), ax=axs[i], **params)
                axs[i].legend()
        else:
            for i, col in enumerate(tqdm(x, desc='Plotting histograms')):
                sns.histplot(data=data, x=col, ax=axs[i], **params)
        plt.tight_layout()
        plt.show()

# Function 4: KDE Plots
def kdeplots(data, shape: tuple, y: str = None, figsize: tuple = (20, 10), params: dict = {}):
    """
    Generates kernel density estimation (KDE) plots for numerical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        shape (tuple): Grid shape for subplots.
        y (str): Optional column for grouping (default: None).
        figsize (tuple): Figure size for the plots (default: (20, 10)).
        params (dict): Additional parameters for sns.kdeplot.
    """
    x = [col for col in data.columns if col != y]
    assert len(x) <= shape[0] * shape[1], "Grid shape is too small for the number of columns."
    
    with load_theme("minimal_light"):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        
        if y:
            for i, col in enumerate(tqdm(x, desc=f'Plotting KDE plots grouped by {y}')):
                for cat in data[y].unique():
                    sns.kdeplot(data[data[y] == cat][col], label=str(cat), ax=axs[i], **params)
                axs[i].legend()
        else:
            for i, col in enumerate(tqdm(x, desc='Plotting KDE plots')):
                sns.kdeplot(data=data, x=col, ax=axs[i], **params)
        plt.tight_layout()
        plt.show()

# Function 5: Box Plots
def boxplots(data, y: str = None, shape: tuple = (1, 2), figsize: tuple = (10, 10), params: dict = {}):
    """
    Generates box plots for numerical columns, optionally grouped by another column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        y (str): Optional column for grouping (default: None).
        shape (tuple): Grid shape for subplots (default: (1, 2)).
        figsize (tuple): Figure size for the plots (default: (10, 10)).
        params (dict): Additional parameters for sns.boxplot.
    """
    x = [col for col in data.columns if col != y]
    assert len(x) <= shape[0] * shape[1], "Grid shape is too small for the number of columns."
    
    with load_theme("arctic_light"):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        for i, col in enumerate(tqdm(x, desc='Generating boxplots')):
            if y:
                sns.boxplot(data=data, x=data[y], y=data[col], ax=axs[i], **params)
            else:
                sns.boxplot(data=data, y=data[col], ax=axs[i], **params)
            axs[i].set_title(col)
        plt.tight_layout()
        plt.show()

# Function 6: Scatter Plots
def scatter_plots(data, x: list, y: str, shape: tuple, figsize: tuple, theme: str = 'minimal_light'):
    """
    Generates scatter plots for combinations of numerical columns, grouped by a categorical column.

    Args:
        data (pd.DataFrame): The input DataFrame.
        x (list): List of numerical columns for scatter plot combinations.
        y (str): Column used for grouping.
        shape (tuple): Grid shape for subplots.
        figsize (tuple): Figure size for the plots.
        theme (str): Theme to be applied using aquarel (default: 'minimal_light').
    """
    combinations = list(itertools.combinations(x, 2))
    assert len(combinations) <= shape[0] * shape[1], "Grid shape is too small for the number of combinations."
    
    with load_theme(theme):
        fig, axs = plt.subplots(*shape, figsize=figsize)
        axs = axs.flatten() if shape != (1, 1) else [axs]
        for i, comb in enumerate(combinations):
            groups = data.groupby(y)[list(comb)]
            for name, group in groups:
                axs[i].plot(group[comb[0]], group[comb[1]], label=name, marker='o', linestyle='', ms=2)
            axs[i].set_xlabel(comb[0])
            axs[i].set_ylabel(comb[1])
            axs[i].legend()
        plt.tight_layout()
        plt.show()
