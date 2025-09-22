from typing import Optional
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature

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
    log_plots: bool = True,
    start_run: bool = True,
    inverse_hue: bool = False,
):
    """
    Logs the classifier run to MLflow.
    """
    categorical_columns = []
    for column in X.columns:
        if X[column].dtype == 'category':
            categorical_columns.append(column)

    df = pd.DataFrame(X.copy())
    df['label'] = y
    if log_plots:
        correlation_plot = plot_correlation_with_label(df, label_column='label')
        feature_importances = plot_feature_importance(model)
        preds = model.predict(X)
        residuals = plot_residuals(preds, y)
        if len(categorical_columns) == 0:
            shap_summary = plot_shap_summary(model, X)
            shap_feature_importances = plot_shap_feature_importance(model, X)
        else:
            shap_summary = None
            shap_feature_importances = None
            logger.warning("Categorical columns found, skipping SHAP summary and feature importances")
        precision_recall_curve = plot_precision_recall_curve(model, X, y)
        confusion_matrix = plot_confusion_matrix(model, X, y)
        model_scores_distribution = plot_model_scores_distribution(model, X, y)
        violin_plot = plot_violin_plot(model, X, y, hue=hue, inverse_hue=inverse_hue)

    if start_run:
        with mlflow.start_run(
            nested=nested, experiment_id=experiment_id, run_name=run_name
        ):
            # Log to MLflow
            mlflow.log_params(params)
            metrics_dict = metrics.model_dump()
            classification_report = metrics_dict.pop('classification_report')
            mlflow.log_metrics(metrics_dict)
            mlflow.log_text(classification_report, "classification_report.txt")
            if log_plots:
                mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")
                mlflow.log_figure(
                    figure=feature_importances, artifact_file="feature_importances.png"
                )
                mlflow.log_figure(figure=residuals, artifact_file="residuals.png")
                if shap_summary:
                    mlflow.log_figure(figure=shap_summary, artifact_file="shap_summary.png")
                if shap_feature_importances:
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

            if log_tags:
                mlflow.set_tags(log_tags)

            if log_model:
                signature = infer_signature(X, preds)
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model",
                    signature=signature,
                    # input_example=X.iloc[:1,:],
                    model_format="ubj",
                    metadata={"feature_set_version": config.feature_set_version},
                )
    else:
        mlflow.log_params(params)
        metrics_dict = metrics.model_dump()
        classification_report = metrics_dict.pop('classification_report')
        mlflow.log_text(classification_report, "classification_report.txt")
        mlflow.log_metrics(metrics_dict)

        mlflow.log_figure(figure=correlation_plot, artifact_file="correlation_plot.png")
        mlflow.log_figure(figure=feature_importances, artifact_file="feature_importances.png")
        mlflow.log_figure(figure=residuals, artifact_file="residuals.png")
        if shap_summary:
            mlflow.log_figure(figure=shap_summary, artifact_file="shap_summary.png")
        if shap_feature_importances:
            mlflow.log_figure(figure=shap_feature_importances, artifact_file="shap_feature_importances.png")
        mlflow.log_figure(figure=precision_recall_curve, artifact_file="precision_recall_curve.png")
        mlflow.log_figure(figure=confusion_matrix, artifact_file="confusion_matrix.png")
        mlflow.log_figure(figure=model_scores_distribution, artifact_file="model_scores_distribution.png")
        mlflow.log_figure(figure=violin_plot, artifact_file="violin_plot.png")
        plt.close()
        if log_tags:
            mlflow.set_tags(log_tags)
        if log_model:
            signature = infer_signature(X, preds)
            mlflow.xgboost.log_model(xgb_model=model, name="model", signature=signature, model_format="ubj", metadata={"feature_set_version": config.feature_set_version})