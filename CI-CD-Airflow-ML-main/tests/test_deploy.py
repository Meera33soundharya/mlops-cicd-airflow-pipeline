import os
import json
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model
from src.predict import predict


def test_model_deployment_and_prediction(tmp_path):
    data_dir = os.path.join(tmp_path, "data")
    model_dir = os.path.join(tmp_path, "models")
    deploy_dir = os.path.join(tmp_path, "production_model")
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
    evaluate_model(
        test_path=res_data["test_path"],
        model_path=model_path,
        metrics_path=metrics_path,
        accuracy_threshold=0.70
    )

    manifest = deploy_model(
        source_model_dir=model_dir,
        target_deploy_dir=deploy_dir,
        metrics_path=metrics_path
    )

    assert os.path.exists(os.path.join(deploy_dir, "model.joblib"))
    assert os.path.exists(os.path.join(deploy_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(deploy_dir, "model_manifest.json"))
    assert manifest["status"] == "DEPLOYED"

    # Test prediction with deployed model
    preds = predict(features=[[5.1, 3.5, 1.4, 0.2]], model_dir=deploy_dir)
    assert len(preds) == 1
    assert "class_id" in preds[0]
    assert "probabilities" in preds[0]
