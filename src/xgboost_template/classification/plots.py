import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.datasets import make_classification
import pandas as pd
import seaborn as sns
import shap


def plot_correlation_with_label(df:pd.DataFrame, label_column:str, save_path=None)->plt.Figure:
  """
  Plots the correlation of each variable in the dataframe with the 'label' column.

  Args:
  - df (pd.DataFrame): DataFrame containing the data, including a 'label' column.
  - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

  Returns:
  - plt.Figure: The plot figure.
  """

  # Compute correlations between all variables and 'label'
  correlations = df.corr()[label_column].drop(label_column).sort_values()

  # Generate a color palette from red to green
  colors = sns.diverging_palette(10, 130, as_cmap=True)
  color_mapped = correlations.map(colors)

  # Set Seaborn style
  sns.set_style(
      "whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5}
  )  # Light grey background and thicker grid lines

  # Create bar plot
  fig = plt.figure(figsize=(12, 8))
  plt.barh(correlations.index, correlations.values, color=color_mapped)

  # Set labels and title with increased font size
  plt.title(f"Correlation with {label_column}", fontsize=18)
  plt.xlabel("Correlation Coefficient", fontsize=16)
  plt.ylabel("Variable", fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(axis="x")

  plt.tight_layout()

  # Save the plot if save_path is specified
  if save_path:
      plt.savefig(save_path, format="png", dpi=600)

  # prevent matplotlib from displaying the chart every time we call this function
  plt.close(fig)

  return fig

def plot_residuals(preds:np.ndarray, y:pd.Series, save_path=None)->plt.Figure:
    """
    Plots the residuals of the model predictions against the true values.

    Args:
    - model: The trained XGBoost model.
    - dvalid (xgb.DMatrix): The validation data in XGBoost DMatrix format.
    - valid_y (pd.Series): The true values for the validation set.
    - save_path (str, optional): Path to save the generated plot. If not specified, plot won't be saved.

    Returns:
    - plt.Figure: The plot figure.
    """

    # Calculate residuals
    residuals = y - preds

    # Set Seaborn style
    sns.set_style("whitegrid", {"axes.facecolor": "#c2c4c2", "grid.linewidth": 1.5})

    # Create scatter plot
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(y, residuals, color="blue", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")

    # Set labels, title and other plot properties
    plt.title("Residuals vs True Values", fontsize=18)
    plt.xlabel("True Values", fontsize=16)
    plt.ylabel("Residuals", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")

    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)

    # Show the plot
    plt.close(fig)

    return fig

def plot_feature_importance(model:xgb.XGBClassifier, save_path=None)->plt.Figure:
    """
    Plots the feature importance of the model.
    """
    # feature_importances = model.feature_importances_
    fig = plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, ax=plt.gca())
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_shap_feature_importance(model:xgb.XGBClassifier, X:pd.DataFrame, save_path=None)->plt.Figure:
    """
    Plots the SHAP summary of the model.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    fig = plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, ax=plt.gca(),show=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_shap_summary(model:xgb.XGBClassifier, X:pd.DataFrame, save_path=None)->plt.Figure:
    """
    Plots the SHAP summary of the model.
    """
    explainer = shap.Explainer(model,X)
    shap_values = explainer(X)
    fig = plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, ax=plt.gca(),plot_size=None,show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_precision_recall_curve(model:xgb.XGBClassifier, X:pd.DataFrame, y:pd.Series, save_path=None)->plt.Figure:
    """
    Plots the precision-recall curve of the model.
    """
    display = PrecisionRecallDisplay.from_estimator(model, X, y)
    fig = plt.figure(figsize=(12, 8))
    display.plot(ax=plt.gca())
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_confusion_matrix(model:xgb.XGBClassifier, X:pd.DataFrame, y:pd.Series, save_path=None)->plt.Figure:
    """
    Plots the confusion matrix of the model.
    """
    fig = plt.figure(figsize=(12, 8))
    display = ConfusionMatrixDisplay.from_estimator(model, X, y)
    display.plot(ax=plt.gca())
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_model_scores_distribution(model:xgb.XGBClassifier, X:pd.DataFrame, y:pd.Series, save_path=None)->plt.Figure:
    """
    Plots the distribution of the model scores.
    """
    fig = plt.figure(figsize=(12, 8))
    # Compute predicted probabilities for the positive class
    scores = model.predict_proba(X)[:, 1]
    # Plot the KDE of scores for each label using seaborn
    for label in sorted(y.unique()):
        sns.kdeplot(scores[y == label], label=f"Label {label}", fill=True, common_norm=False, alpha=0.5)
    plt.title("Model Scores Distribution by Label")
    plt.xlabel("Predicted Probability (Positive Class)")
    plt.ylabel("Density")
    plt.legend(title="True Label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

def plot_violin_plot(model:xgb.XGBClassifier, X:pd.DataFrame, y:pd.Series, hue:Optional[pd.Series]=None, inverse_hue:bool=True, save_path=None)->plt.Figure:
    """
    Plots the violin plot of the model scores.
    Args:
    - model: The trained XGBoost model.
    - X: The input data.
    - y: The target variable, you will get on violin plot in x axis for every label
    - hue: The variable to use for the hue, for every y, you will get multiple hue violin plots to see the distribution of the model scores for each hue, for the same y
    - inverse_hue: If True, x will be the hue and hue will be x
    - save_path: The path to save the plot.
    Returns:
    - plt.Figure: The plot figure.
    """
    fig = plt.figure(figsize=(12, 8))
    scores = model.predict_proba(X)[:, 1]
    # x, hue = hue, scores if inverse_hue else scores, hue
    data = pd.DataFrame({'scores': scores, 'label': y})
    if hue is not None:
        data['hue'] = hue
    x_column, hue_column = ('hue', 'label') if inverse_hue else ('label', 'hue')
    sns.violinplot(x=x_column, y="scores", ax=plt.gca(), hue=hue_column, data=data)
    # sns.stripplot(x=x_column, y="scores", data=data, jitter=True, hue=hue_column, ax=plt.gca())
    plt.title("Model Scores Violin Plot")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="png", dpi=600)
    plt.close(fig)
    return fig

if __name__ == "__main__":
    import xgboost as xgb
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_clusters_per_class=1)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["label"] = y
    # Test the function

    model = xgb.XGBClassifier()
    model.fit(X, y)

    root_save_path = Path("plots")
    root_save_path.mkdir(parents=True, exist_ok=True)
    correlation_plot = plot_correlation_with_label(df, label_column="label", save_path=root_save_path / "correlation_plot.png")
    
    df['category'] = np.random.choice(['a', 'b', 'c'], size=len(df))
    
    residuals_plot = plot_residuals(model.predict(X), df['label'], save_path=root_save_path / "residuals_plot.png")
    feature_importance_plot = plot_feature_importance(model, save_path=root_save_path / "feature_importance_plot.png")
    shap_summary_plot = plot_shap_summary(model, X, save_path=root_save_path / "shap_summary_plot.png")
    precision_recall_curve_plot = plot_precision_recall_curve(model, X, y, save_path=root_save_path / "precision_recall_curve_plot.png")
    confusion_matrix_plot = plot_confusion_matrix(model, X, df['label'], save_path=root_save_path / "confusion_matrix_plot.png")
    shap_feature_importance_plot = plot_shap_feature_importance(model, X, save_path=root_save_path / "shap_feature_importance_plot.png")
    model_scores_distribution_plot = plot_model_scores_distribution(model, X, df['label'], save_path=root_save_path / "model_scores_distribution_plot.png")
    violin_plot = plot_violin_plot(model, X, df['label'], hue=df['category'], save_path=root_save_path / "violin_plot.png")
