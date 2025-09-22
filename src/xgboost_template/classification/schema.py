from pydantic import BaseModel, Field


class OptimizationParams(BaseModel):

    n_trials: int = Field(default=100)
    project_name: str = Field(default="xgboost_template")
    feature_set_version: str = Field(default="1.0.0")
    num_boost_round: int = Field(default=100)
    nfold: int = Field(default=5)
    early_stopping_rounds: int = Field(default=10)

class ClassifierMetrics(BaseModel):
  accuracy: float
  precision: float
  recall: float
  f1: float
  roc_auc: float
  average_precision: float
  classification_report: str

  def __repr__(self):
    return f"ClassifierMetrics(accuracy={self.accuracy}, precision={self.precision}, recall={self.recall}, f1={self.f1}, roc_auc={self.roc_auc}, average_precision={self.average_precision})"

  def __str__(self):
    return f"ClassifierMetrics(accuracy={self.accuracy}, precision={self.precision}, recall={self.recall}, f1={self.f1}, roc_auc={self.roc_auc}, average_precision={self.average_precision})"
