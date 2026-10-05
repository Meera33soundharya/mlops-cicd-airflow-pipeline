import os
import sys
import subprocess


def run_cmd(command: str):
    print(f"\n==================================================")
    print(f"Executing: {command}")
    print(f"==================================================")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    python_bin = sys.executable

    print(">>> STAGE 1: Running Unit Tests (CI)...")
    run_cmd(f'"{python_bin}" -m pytest tests/ -v')

    print("\n>>> STAGE 2: Preprocessing Data...")
    run_cmd(f'"{python_bin}" src/data_preprocessing.py')

    print("\n>>> STAGE 3: Training Model...")
    run_cmd(f'"{python_bin}" src/train.py')

    print("\n>>> STAGE 4: Evaluating Model & Validating Quality Gate...")
    run_cmd(f'"{python_bin}" src/evaluate.py')

    print("\n>>> STAGE 5: Deploying Validated Model...")
    run_cmd(f'"{python_bin}" src/deploy.py')

    print("\n>>> STAGE 6: Running Production Inference Smoke Test...")
    run_cmd(f'"{python_bin}" src/predict.py')

    print("\nSUCCESS: All pipeline stages completed successfully!")


if __name__ == "__main__":
    main()
