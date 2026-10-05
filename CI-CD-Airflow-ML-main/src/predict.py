import os
import joblib
import pandas as pd
import numpy as np


def predict(features: list, model_dir: str = "production_model") -> list:
    """
    Runs prediction using deployed production model and scaler.
    :param features: 2D list or list of feature values [[sepal_len, sepal_width, petal_len, petal_width]]
    :param model_dir: Path to deployment directory containing model.joblib and scaler.joblib
    """
    model_path = os.path.join(model_dir, "model.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Deployed model artifacts missing in {model_dir}. Ensure model is deployed.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features_array = np.array(features)
    if len(features_array.shape) == 1:
        features_array = features_array.reshape(1, -1)

    if hasattr(scaler, "feature_names_in_"):
        features_df = pd.DataFrame(features_array, columns=scaler.feature_names_in_)
    else:
        features_df = features_array

    scaled_features = scaler.transform(features_df)
    if hasattr(scaler, "feature_names_in_"):
        scaled_df = pd.DataFrame(scaled_features, columns=scaler.feature_names_in_)
    else:
        scaled_df = scaled_features

    predictions = model.predict(scaled_df)
    probabilities = model.predict_proba(scaled_df)

    return [
        {"class_id": int(pred), "probabilities": prob.tolist()}
        for pred, prob in zip(predictions, probabilities)
    ]


if __name__ == "__main__":
    sample_input = [[5.1, 3.5, 1.4, 0.2]]
    output = predict(sample_input)
    print(f"Sample Input: {sample_input}")
    print(f"Prediction Output: {output}")
