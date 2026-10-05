import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_model(
    train_path: str = "data/processed/train.csv",
    model_output_path: str = "models/model.joblib",
    metadata_output_path: str = "models/training_metadata.json"
) -> dict:
    """
    Trains a machine learning model using preprocessed training data,
    saves the trained model artifact and training metadata.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run preprocessing first.")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    n_estimators = 100
    max_depth = 5
    random_state = 42

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_output_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "num_samples": len(X_train),
        "num_features": X_train.shape[1],
        "feature_names": list(X_train.columns)
    }

    with open(metadata_output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model trained successfully and saved to {model_output_path}")
    return metadata


if __name__ == "__main__":
    train_model()
