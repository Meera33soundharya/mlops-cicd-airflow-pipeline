# Automated CI/CD ML Pipeline with GitHub Actions and Apache Airflow

A complete, production-grade MLOps repository demonstrating automated **Model Testing**, **Training**, **Validation Performance Gating**, and **Deployment** triggered on every code commit via **GitHub Actions** and orchestrated with **Apache Airflow**.

---

## 🏗️ Architecture & Workflow

```
                        +----------------------------+
                        |  Developer Push / Commit   |
                        +--------------+-------------+
                                       |
                                       v
                    +------------------------------------+
                    |   GitHub Actions CI/CD Workflow    |
                    |   (.github/workflows/ml_cicd.yml)  |
                    +------------------+-----------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      [ Stage 1: Unit Testing ]              [ Stage 2: Airflow DAG / CD ]
      - Data schema checks                   - Preprocess Data
      - Model training integrity             - Train RandomForest Classifier
      - Quality gate logic                   - Evaluate & Gate (Accuracy >= 0.85)
      - Deployment manifest tests            - Deploy to Production Directory
                                             - Artifact Upload (GitHub Actions)
```

---

## 📁 Repository Structure

```
CI-CD-Airflow-ML/
├── .github/
│   └── workflows/
│       └── ml_cicd.yml           # GitHub Actions CI/CD Pipeline configuration
├── dags/
│   └── ml_pipeline_dag.py        # Apache Airflow DAG defining task execution order
├── src/
│   ├── data_preprocessing.py     # Dataset loading, cleaning, feature scaling, & split
│   ├── train.py                  # ML model training & hyperparameter metadata output
│   ├── evaluate.py               # Model evaluation & performance threshold quality gate
│   ├── deploy.py                 # Model artifact promotion & production manifest generation
│   └── predict.py                # Deployed model inference service
├── tests/
│   ├── test_data.py              # Unit tests for preprocessing & dataset shapes
│   ├── test_train.py             # Unit tests for training logic & artifact output
│   ├── test_evaluate.py          # Unit tests for quality gate evaluation
│   └── test_deploy.py            # Unit tests for deployment promotion & inference
├── data/
│   └── processed/                # Processed train/test data CSV files
├── models/                       # Staging directory for model & scaler artifacts
├── production_model/             # Deployment directory containing active production model
├── requirements.txt              # Project Python dependencies
├── run_pipeline.py               # Local end-to-end pipeline execution runner
└── README.md                     # Technical documentation
```

---

## 🚀 Key Features

1. **Continuous Integration (CI)**:
   - Automated unit test suite (`pytest`) testing data integrity, training stability, quality gate logic, and deployment manifests before any deployment step.
2. **Apache Airflow DAG Orchestration**:
   - `ml_pipeline_dag` organizes execution into 4 sequential tasks:
     `preprocess_data_task` $\rightarrow$ `train_model_task` $\rightarrow$ `evaluate_model_task` $\rightarrow$ `deploy_model_task`.
3. **Automated Quality Gate**:
   - Evaluation stage checks accuracy against a minimum threshold ($\ge 0.85$). If the model underperforms, the pipeline fails and halts deployment.
4. **Automated Deployment & Versioning**:
   - Promotes validated model artifacts (`model.joblib`, `scaler.joblib`, `metrics.json`) to `production_model/` and writes `model_manifest.json` with timestamp and execution metrics.
5. **GitHub Actions Automation**:
   - Executes automatically on `push` or `pull_request` to `main`/`master` branches and archives trained model artifacts for each build run.

---

## 💻 How to Run Locally

### 1. Prerequisites & Virtual Environment Setup

Ensure Python 3.11+ is installed.

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Execute Unit Tests

```bash
pytest tests/ -v
```

### 3. Run End-to-End Pipeline

Execute all stages locally from preprocessing to inference testing:

```bash
python run_pipeline.py
```

### 4. Run Model Inference

Make predictions using the deployed production model:

```bash
python src/predict.py
```

---

## 💨 Apache Airflow Setup (Optional Local Airflow)

To test the Airflow DAG locally:

```bash
# Initialize Airflow DB
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname MLOps \
    --lastname Admin \
    --role Admin \
    --email admin@example.com

# Test DAG tasks individually
airflow tasks test ml_pipeline_dag preprocess_data_task 2026-01-01
airflow tasks test ml_pipeline_dag train_model_task 2026-01-01
airflow tasks test ml_pipeline_dag evaluate_model_task 2026-01-01
airflow tasks test ml_pipeline_dag deploy_model_task 2026-01-01
```

---

## 🛡️ Quality Gate & Deployment Manifest

### Sample `models/metrics.json`
```json
{
    "accuracy": 0.9333,
    "precision": 0.9333,
    "recall": 0.9333,
    "f1_score": 0.9333,
    "accuracy_threshold": 0.85,
    "passed_quality_gate": true
}
```

### Sample `production_model/model_manifest.json`
```json
{
    "status": "DEPLOYED",
    "deployment_timestamp": "2026-09-18T07:18:31Z",
    "version": "v1.0.0",
    "deployed_files": [
        "model.joblib",
        "scaler.joblib",
        "metrics.json",
        "training_metadata.json"
    ],
    "metrics": {
        "accuracy": 0.9333,
        "passed_quality_gate": true
    }
}
```
"# CI-CD-Airflow-ML" 
