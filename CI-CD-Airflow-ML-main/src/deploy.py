import os
import shutil
import json
import time


def deploy_model(
    source_model_dir: str = "models",
    target_deploy_dir: str = "production_model",
    metrics_path: str = "models/metrics.json"
) -> dict:
    """
    Promotes validated model artifacts to production deployment directory and writes deployment manifest.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("Metrics file missing. Evaluation must complete before deployment.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if not metrics.get("passed_quality_gate", False):
        raise ValueError("Cannot deploy model: Model did not pass quality gate threshold!")

    os.makedirs(target_deploy_dir, exist_ok=True)

    artifacts_to_copy = ["model.joblib", "scaler.joblib", "metrics.json", "training_metadata.json"]
    copied_files = []

    for file_name in artifacts_to_copy:
        src = os.path.join(source_model_dir, file_name)
        dst = os.path.join(target_deploy_dir, file_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_files.append(file_name)

    manifest = {
        "status": "DEPLOYED",
        "deployment_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": "v1.0.0",
        "deployed_files": copied_files,
        "metrics": metrics
    }

    manifest_path = os.path.join(target_deploy_dir, "model_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Model successfully deployed to '{target_deploy_dir}'. Deployed files: {copied_files}")
    return manifest


if __name__ == "__main__":
    deploy_model()
