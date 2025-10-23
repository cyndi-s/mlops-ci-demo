import argparse, os, sys, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.csv")
    p.add_argument("--out", default="model.pkl")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    a = p.parse_args()

    df = pd.read_csv(a.data)
    needed = {"temp","cough_score","label"}
    if not needed.issubset(df.columns):
        print(f"ERROR: data must have {sorted(needed)}", file=sys.stderr)
        sys.exit(1)

    X = df[["temp","cough_score"]].values
    y = df["label"].astype(int).values
    Xtr,Xte,ytr,yte = train_test_split(
        X,y,test_size=a.test_size,random_state=a.random_state,stratify=y)

    m = LogisticRegression(solver="liblinear",random_state=a.random_state)
    m.fit(Xtr,ytr)
    yp = m.predict(Xte)

    acc,prec,rec = accuracy_score(yte,yp),precision_score(yte,yp),recall_score(yte,yp)
    joblib.dump(m,a.out)

    print(f"rows={len(df)}")
    print(f"test_size={a.test_size}")
    print(f"accuracy={acc:.6f}")
    print(f"precision={prec:.6f}")
    print(f"recall={rec:.6f}")
    print(f"model_path={os.path.abspath(a.out)}")

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_experiment("mlops-ci-demo")
        mlflow.sklearn.autolog(log_models=True)

        # train inside autolog context
        m = LogisticRegression(solver="liblinear", random_state=a.random_state)
        m.fit(Xtr, ytr)

        yp = m.predict(Xte)
        acc = accuracy_score(yte, yp)
        prec = precision_score(yte, yp)
        rec = recall_score(yte, yp)

        joblib.dump(m, a.out)

        print(f"rows={len(df)}")
        print(f"test_size={a.test_size}")
        print(f"accuracy={acc:.6f}")
        print(f"precision={prec:.6f}")
        print(f"recall={rec:.6f}")
        print(f"model_path={os.path.abspath(a.out)}")

    except Exception as e:
        print(f"mlflow_logging_skipped={e.__class__.__name__}")

if __name__=="__main__":
    main()
