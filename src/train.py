# src/train_dataset2.py
import argparse, os, sys
import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

# Optional fallback for local demos (ignored in CI when env vars are set)
if not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("file:./mlruns")

# Use your demo experiment name
mlflow.set_experiment("mlops-ci-demo")

# Autolog creates a CHILD run for .fit() with rich params/metrics/artifacts
mlflow.sklearn.autolog(log_models=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--out", default="model.pkl")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    a = p.parse_args()

    # Load & validate data
    df = pd.read_csv(a.data)
    needed = {"temp", "cough_score", "label"}
    if not needed.issubset(df.columns):
        print(f"ERROR: data must have {sorted(needed)}", file=sys.stderr)
        sys.exit(1)

    X = df[["temp", "cough_score"]].values
    y = df["label"].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=a.test_size, random_state=a.random_state, stratify=y
    )

    # Train (sklearn autolog will create/log a child run here)
    model = LogisticRegression(solver="liblinear", random_state=a.random_state)
    model.fit(Xtr, ytr)

    # Evaluate
    yp = model.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec = precision_score(yte, yp)
    rec = recall_score(yte, yp)

    # Save model file
    joblib.dump(model, a.out)

    # ---- Parent-run summary so the TOP run isn't empty on DagsHub ----
    # (MLflow Projects made a parent run; these go to that run.)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("test_size", float(a.test_size))
    mlflow.log_param("random_state", int(a.random_state))

    mlflow.log_metric("accuracy_top", float(acc))
    mlflow.log_metric("precision_top", float(prec))
    mlflow.log_metric("recall_top", float(rec))

    # Surface key artifacts at the parent level, too
    mlflow.log_artifact(a.data)
    if os.path.exists(a.out):
        mlflow.log_artifact(a.out)
    # ------------------------------------------------------------------

    # Console prints (nice for GitHub Actions log)
    print(f"rows={len(df)}")
    print(f"test_size={a.test_size}")
    print(f"accuracy={acc:.6f}")
    print(f"precision={prec:.6f}")
    print(f"recall={rec:.6f}")
    print(f"model_path={os.path.abspath(a.out)}")

if __name__ == "__main__":
    main()
