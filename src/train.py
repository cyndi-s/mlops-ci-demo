import argparse, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")  # headless backend for CI/CD
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mlflow



p=argparse.ArgumentParser(); 
p.add_argument("--data",default="data.csv")
p.add_argument("--out",default="model.pkl")
a=p.parse_args()

df=pd.read_csv(a.data)
X=df[['x']].values
y=df['y'].values
m=LinearRegression().fit(X,y)
print("coef:",m.coef_[0],"intercept:",m.intercept_)


# Plot & log
y_pred = m.predict(X)
plt.scatter(X, y, s = 18, label="data")
plt.plot(X, y_pred, color="orange", label="fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("toy-ml regression fit")
plt.legend()
plt.tight_layout()
plt.savefig("plot.png", dpi=150)
mlflow.log_artifact("plot.png")