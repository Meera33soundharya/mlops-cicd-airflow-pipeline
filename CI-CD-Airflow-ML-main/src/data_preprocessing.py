import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data_dir: str = "data/processed", models_dir: str = "models") -> dict:
    """
    Loads raw dataset, performs feature scaling and train/test split,
    and saves processed data and scaler artifacts.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save artifacts
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_data["target"] = y_train.values

    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data["target"] = y_test.values

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    scaler_path = os.path.join(models_dir, "scaler.joblib")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print(f"Data preprocessed successfully. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    return {
        "train_path": train_path,
        "test_path": test_path,
        "scaler_path": scaler_path
    }


if __name__ == "__main__":
    preprocess_data()
