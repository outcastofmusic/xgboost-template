from typing import Optional

import mlflow
import pandas as pd
import xgboost as xgb

from .plots import (
    plot_confusion_matrix,
    plot_correlation_with_label,
    plot_feature_importance,
    plot_model_scores_distribution,
    plot_precision_recall_curve,
    plot_residuals,
    plot_shap_feature_importance,
    plot_shap_summary,
    plot_violin_plot,
)
from .schema import ClassifierMetrics, OptimizationParams


def log_classifier_run(
    model: xgb.XGBClassifier,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    config: OptimizationParams,
    metrics: ClassifierMetrics,
    hue: Optional[pd.Series] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    nested: bool = True,
    log_tags: Optional[dict] = None,
    log_model: bool = True,
    start_run: bool = True,
):
    """
    Logs the classifier run to MLflow.
    """
    correlation_plot = plot_correlation_with_label(X, y)
    feature_importances = plot_feature_importance(model, booster=params.get("booster"))
    residuals = plot_residuals(model, X, y)
    shap_summary = plot_shap_summary(model, X)
    shap_feature_importances = plot_shap_feature_importance(model, X)
    precision_recall_curve = plot_precision_recall_curve(model, X, y)
    confusion_matrix = plot_confusion_matrix(model, X, y)
    model_scores_distribution = plot_model_scores_distribution(model, X, y)
    violin_plot = plot_violin_plot(model, X, y, hue=hue)

    if start_run:
        with mlflow.start_run(
            nested=nested, experiment_id=experiment_id, run_name=run_name
        ):
            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_metrics(metrics.model_dump())

            mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")
            mlflow.log_figure(
                figure=feature_importances, artifact_file="feature_importances.png"
            )
            mlflow.log_figure(figure=residuals, artifact_file="residuals.png")
            mlflow.log_figure(figure=shap_summary, artifact_file="shap_summary.png")
            mlflow.log_figure(
                figure=shap_feature_importances,
                artifact_file="shap_feature_importances.png",
            )
            mlflow.log_figure(
                figure=precision_recall_curve, artifact_file="precision_recall_curve.png"
            )
            mlflow.log_figure(figure=confusion_matrix, artifact_file="confusion_matrix.png")
            mlflow.log_figure(
                figure=model_scores_distribution,
                artifact_file="model_scores_distribution.png",
            )
            mlflow.log_figure(figure=violin_plot, artifact_file="violin_plot.png")

            mlflow.set_tags(log_tags)

            if log_model:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                    input_example=X.iloc[[0]],
                    model_format="ubj",
                    metadata={"feature_set_version": config.feature_set_version},
                )
    else:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics.model_dump())

        mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")
        mlflow.log_figure(figure=feature_importances, artifact_file="feature_importances.png")
        mlflow.log_figure(figure=residuals, artifact_file="residuals.png")
        mlflow.log_figure(figure=shap_summary, artifact_file="shap_summary.png")
        mlflow.log_figure(figure=shap_feature_importances, artifact_file="shap_feature_importances.png")
        mlflow.log_figure(figure=precision_recall_curve, artifact_file="precision_recall_curve.png")
        mlflow.log_figure(figure=confusion_matrix, artifact_file="confusion_matrix.png")
        mlflow.log_figure(figure=model_scores_distribution, artifact_file="model_scores_distribution.png")
        mlflow.log_figure(figure=violin_plot, artifact_file="violin_plot.png")
        mlflow.set_tags(log_tags)
        if log_model:
            mlflow.xgboost.log_model(xgb_model=model, name="model", input_example=X.iloc[[0]], model_format="ubj", metadata={"feature_set_version": config.feature_set_version})