import os
import sys
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(
    test_path: str = "data/processed/test.csv",
    model_path: str = "models/model.joblib",
    metrics_path: str = "models/metrics.json",
    accuracy_threshold: float = 0.85
) -> dict:
    """
    Evaluates the trained model against test data and checks performance against quality threshold.
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro"))
    rec = float(recall_score(y_test, y_pred, average="macro"))
    f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    passed_quality_gate = acc >= accuracy_threshold

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "accuracy_threshold": accuracy_threshold,
        "passed_quality_gate": passed_quality_gate
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model Evaluation Metrics:")
    print(f" - Accuracy:  {acc:.4f} (Threshold: {accuracy_threshold})")
    print(f" - Precision: {prec:.4f}")
    print(f" - Recall:    {rec:.4f}")
    print(f" - F1 Score:  {f1:.4f}")
    print(f" - Quality Gate Status: {'PASSED' if passed_quality_gate else 'FAILED'}")

    return metrics


if __name__ == "__main__":
    res = evaluate_model()
    if not res["passed_quality_gate"]:
        print("Error: Model failed validation quality gate threshold.")
        sys.exit(1)
