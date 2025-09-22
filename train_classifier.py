from xgboost_template import get_or_create_experiment
from sklearn.datasets import make_classification
import mlflow
import numpy as np
import pandas as pd
from xgboost_template.classification import optimize_classifier, OptimizationParams
from sklearn.preprocessing import LabelEncoder
import warnings
# warnings.simplefilter("error")

mlflow.set_tracking_uri("http://127.0.0.1:8081")

X,Y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, n_clusters_per_class=1)


dataset = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
dataset["target"] = Y
dataset['split'] = np.random.choice(['train', 'valid', 'test'], size=len(dataset), p=[0.8, 0.1, 0.1])
dataset['stratified'] = np.random.choice(['a', 'b', 'c'], size=len(dataset))
dataset['categorical'] = np.random.choice(['a', 'b', 'c'], size=len(dataset))
label_encoder = LabelEncoder()
dataset['categorical'] = label_encoder.fit_transform(dataset['categorical'])
dataset['categorical'] = dataset['categorical'].astype('category')

print(dataset.head())
experiment_id = get_or_create_experiment(experiment_name="xgboost_template")

optimization_params = OptimizationParams(n_trials=5)
optimize_classifier(
    dataset=dataset,
    target_column="target",
    experiment_id=experiment_id,
    run_name="xgboost_template",
    config=optimization_params,
    categorical_columns=["categorical"],
    stratified_column="stratified",
)