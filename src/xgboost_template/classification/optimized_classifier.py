import mlflow
import optuna
import pandas as pd
from typing import Optional
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

import mlflow
from .schema import OptimizationParams, ClassifierMetrics
from .experiment_logger import log_classifier_run


# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


def evaluate_metrics(
    model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series
) -> ClassifierMetrics:
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    average_precision = average_precision_score(y, preds)
    classification_report_output = classification_report(y, preds)
    return ClassifierMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        average_precision=average_precision,
        classification_report=classification_report_output,
    )


def optimize_classifier(
    dataset: pd.DataFrame,
    target_column: str,
    experiment_id: str,
    run_name: str,
    config: OptimizationParams,
    categorical_columns: Optional[list[str]] = None,
    random_state: Optional[int] = None,
    stratified_column: Optional[str] = None,
):

    if categorical_columns:
        for category in categorical_columns:
            dataset[category] = dataset[category].astype("category")
    if stratified_column:
        dataset[stratified_column] = dataset[stratified_column].astype("category")

    train_x = pd.DataFrame(dataset.loc[dataset.split == "train"].drop(columns=[target_column, "split"]))
    if stratified_column:
        train_x_stratified = train_x[stratified_column]
        train_x = train_x.drop(columns=[stratified_column])
    else:
        train_x_stratified = None
    train_y = pd.Series(dataset.loc[dataset.split == "train"][target_column])
    valid_x = pd.DataFrame(dataset.loc[dataset.split == "valid"].drop(columns=[target_column, "split"]))
    if stratified_column:
        valid_x_stratified = valid_x[stratified_column]
        valid_x = valid_x.drop(columns=[stratified_column])
    else:
        valid_x_stratified = None
    valid_y = pd.Series(dataset.loc[dataset.split == "valid"][target_column])
    test_x = pd.DataFrame(dataset.loc[dataset.split == "test"].drop(columns=[target_column, "split"]))
    if stratified_column:
        test_x_stratified = test_x[stratified_column]
        test_x = test_x.drop(columns=[stratified_column])
    else:
        test_x_stratified = None

    test_y = pd.Series(dataset.loc[dataset.split == "test"][target_column])

    dtrain = xgb.DMatrix(train_x, label=train_y, enable_categorical=True if categorical_columns else False)
    # dvalid = xgb.DMatrix(valid_x, label=valid_y, enable_categorical=True if categorical_columns else False)
    # dtest = xgb.DMatrix(test_x, label=test_y, enable_categorical=True if categorical_columns else False)

    def objective(trial):
        # Define hyperparameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        }

        if params["booster"] == "gbtree" or params["booster"] == "dart":
            params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            params["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )
            params['enable_categorical'] = True if categorical_columns else False

        # Train XGBoost model
        # cv_results = xgb.cv(
        #     params,
        #     dtrain=dtrain,
        #     num_boost_round=config.num_boost_round,
        #     nfold=config.nfold,
        #     early_stopping_rounds=config.early_stopping_rounds,
        #     maximize=True,
        #     metrics=["logloss", "auc"],
        #     as_pandas=True,
        # )
        model = xgb.train(params, dtrain, num_boost_round=config.num_boost_round, early_stopping_rounds=config.early_stopping_rounds)
        metrics = evaluate_metrics(model, valid_x, valid_y)
        log_classifier_run(
            model=model,
            params=params,
            X=valid_x,
            y=valid_y,
            config=config,
            metrics=metrics,
            hue=valid_x_stratified,
            experiment_id=None,
            run_name=None,
            nested=True,
            log_tags=None,
            log_model=False,
        )
        return metrics.average_precision

    # Initiate the parent run and call the hyperparameter tuning child run logic
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # Initialize the Optuna study
        study = optuna.create_study(direction="maximize")

        # Execute the hyperparameter optimization trials.
        # Note the addition of the `champion_callback` inclusion to control our logging
        study.optimize(
            objective, n_trials=config.n_trials, callbacks=[champion_callback]
        )

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_average_precision", study.best_value)

        tags = {
            "project": config.project_name,
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": config.feature_set_version,
        }

        # Log a fit model instance
        model = xgb.train(study.best_params, dtrain, num_boost_round=config.num_boost_round, early_stopping_rounds=config.early_stopping_rounds)

        metrics = evaluate_metrics(model, test_x, test_y)
        log_classifier_run(
            model=model,
            params=study.best_params,
            X=test_x,
            y=test_y,
            config=config,
            metrics=metrics,
            hue=test_x_stratified,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=True,
            log_tags=tags,
            log_model=True,
            start_run=False,
        )
