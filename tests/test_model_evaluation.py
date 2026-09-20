"""Tests for model evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fraud_detection.model_evaluation import ModelEvaluator


def _tiny_binary_dataset():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "amount": rng.normal(size=40),
            "hour": rng.integers(0, 24, size=40),
        }
    )
    y = pd.Series((X["amount"] > 0).astype(int))
    return X, y


def test_shap_values_support_logistic_regression():
    X, y = _tiny_binary_dataset()
    model = LogisticRegression(max_iter=200).fit(X, y)

    shap_values = ModelEvaluator._shap_values_for_plot(model, X)

    assert shap_values.shape == X.shape


def test_shap_values_support_tree_models():
    X, y = _tiny_binary_dataset()
    model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

    shap_values = ModelEvaluator._shap_values_for_plot(model, X)

    assert shap_values.shape[0] == len(X)
    assert shap_values.shape[1] == X.shape[1]
