import os
import json
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def test_model_evaluation(tmp_path):
    data_dir = os.path.join(tmp_path, "data")
    model_dir = os.path.join(tmp_path, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")
    metrics_path = os.path.join(model_dir, "metrics.json")

    res_data = preprocess_data(data_dir=data_dir, models_dir=model_dir)
    train_model(
        train_path=res_data["train_path"],
        model_output_path=model_path,
        metadata_output_path=metadata_path
    )

    metrics = evaluate_model(
        test_path=res_data["test_path"],
        model_path=model_path,
        metrics_path=metrics_path,
        accuracy_threshold=0.70
    )

    assert os.path.exists(metrics_path)
    assert "accuracy" in metrics
    assert "passed_quality_gate" in metrics
    assert metrics["passed_quality_gate"] is True
    assert metrics["accuracy"] >= 0.70
