from pydantic import BaseModel, Field
from typing import List
import numpy as np


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

  def get_numeric_metrics(self) -> dict:
    """Extract numeric metrics as a dictionary, excluding classification_report."""
    return {
      'accuracy': self.accuracy,
      'precision': self.precision,
      'recall': self.recall,
      'f1': self.f1,
      'roc_auc': self.roc_auc,
      'average_precision': self.average_precision
    }

  @staticmethod
  def calculate_means(metrics_list: List['ClassifierMetrics']) -> 'ClassifierMetrics':
    """
    Calculate the mean of all numeric metrics from a list of ClassifierMetrics.
    
    Args:
        metrics_list: List of ClassifierMetrics objects to calculate means from
        
    Returns:
        ClassifierMetrics object with mean values and combined classification report
    """
    if not metrics_list:
      raise ValueError("Cannot calculate means from empty list")
    
    # Extract all numeric values for each metric as numpy arrays
    metric_values = {
      'accuracy': np.array([m.accuracy for m in metrics_list]),
      'precision': np.array([m.precision for m in metrics_list]),
      'recall': np.array([m.recall for m in metrics_list]),
      'f1': np.array([m.f1 for m in metrics_list]),
      'roc_auc': np.array([m.roc_auc for m in metrics_list]),
      'average_precision': np.array([m.average_precision for m in metrics_list])
    }
    
    # Calculate means
    means = {metric: float(np.mean(values)) for metric, values in metric_values.items()}
    
    # Combine classification reports
    combined_report = f"Mean metrics from {len(metrics_list)} runs:\n"
    combined_report += "\n".join([f"Run {i+1}:\n{m.classification_report}" for i, m in enumerate(metrics_list)])
    
    return ClassifierMetrics(
      accuracy=means['accuracy'],
      precision=means['precision'],
      recall=means['recall'],
      f1=means['f1'],
      roc_auc=means['roc_auc'],
      average_precision=means['average_precision'],
      classification_report=combined_report
    )

  @staticmethod
  def calculate_stds(metrics_list: List['ClassifierMetrics']) -> 'ClassifierMetrics':
    """
    Calculate the standard deviation of all numeric metrics from a list of ClassifierMetrics.
    
    Args:
        metrics_list: List of ClassifierMetrics objects to calculate standard deviations from
        
    Returns:
        ClassifierMetrics object with standard deviation values
    """
    if not metrics_list:
      raise ValueError("Cannot calculate standard deviations from empty list")
    
    if len(metrics_list) < 2:
      raise ValueError("Cannot calculate standard deviations from less than 2 samples")
    
    # Extract all numeric values for each metric as numpy arrays
    metric_values = {
      'accuracy': np.array([m.accuracy for m in metrics_list]),
      'precision': np.array([m.precision for m in metrics_list]),
      'recall': np.array([m.recall for m in metrics_list]),
      'f1': np.array([m.f1 for m in metrics_list]),
      'roc_auc': np.array([m.roc_auc for m in metrics_list]),
      'average_precision': np.array([m.average_precision for m in metrics_list])
    }
    
    # Calculate standard deviations
    stds = {metric: float(np.std(values, ddof=1)) for metric, values in metric_values.items()}
    
    # Create report for standard deviations
    std_report = f"Standard deviations from {len(metrics_list)} runs:\n"
    std_report += f"Accuracy: {stds['accuracy']:.6f}\n"
    std_report += f"Precision: {stds['precision']:.6f}\n"
    std_report += f"Recall: {stds['recall']:.6f}\n"
    std_report += f"F1: {stds['f1']:.6f}\n"
    std_report += f"ROC AUC: {stds['roc_auc']:.6f}\n"
    std_report += f"Average Precision: {stds['average_precision']:.6f}"
    
    return ClassifierMetrics(
      accuracy=stds['accuracy'],
      precision=stds['precision'],
      recall=stds['recall'],
      f1=stds['f1'],
      roc_auc=stds['roc_auc'],
      average_precision=stds['average_precision'],
      classification_report=std_report
    )

  @staticmethod
  def calculate_means_and_stds(metrics_list: List['ClassifierMetrics']) -> tuple['ClassifierMetrics', 'ClassifierMetrics']:
    """
    Calculate both means and standard deviations from a list of ClassifierMetrics.
    
    Args:
        metrics_list: List of ClassifierMetrics objects
        
    Returns:
        Tuple of (mean_metrics, std_metrics)
    """
    return ClassifierMetrics.calculate_means(metrics_list), ClassifierMetrics.calculate_stds(metrics_list)
