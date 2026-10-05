import os
import joblib
from src.data_preprocessing import preprocess_data
from src.train import train_model


def test_model_training(tmp_path):
    data_dir = os.path.join(tmp_path, "data")
    model_dir = os.path.join(tmp_path, "models")
    model_path = os.path.join(model_dir, "model.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    res_data = preprocess_data(data_dir=data_dir, models_dir=model_dir)
    metadata = train_model(
        train_path=res_data["train_path"],
        model_output_path=model_path,
        metadata_output_path=metadata_path
    )

    assert os.path.exists(model_path)
    assert os.path.exists(metadata_path)

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
    assert metadata["model_type"] == "RandomForestClassifier"
