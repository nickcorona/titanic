from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder  # sometimes needed
from dirty_cat import SimilarityEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

from helpers import encode_dates, loguniform

df = pd.read_csv(
    r"data/train.csv",
    parse_dates=[],
    index_col=[],
)

print(
    pd.concat([df.dtypes, df.nunique() / len(df)], axis=1)
    .rename({0: "dtype", 1: "proportion unique"}, axis=1)
    .sort_values(["dtype", "proportion unique"])
)

y = df["Survived"]
X = df.drop(
    [
        "Survived",
    ],
    axis=1,
)

X.info()

encode_columns = ["Cabin", "Ticket", "Name"]
enc = SimilarityEncoder(similarity="ngram", categories="k-means", n_prototypes=5)
for col in encode_columns:
    transformed_values = enc.fit_transform(X[col].values.reshape(-1, 1))
    transformed_values = pd.DataFrame(transformed_values, index=X.index)
    transformed_values.columns = [f"{col}_" + str(num) for num in transformed_values]
    X = pd.concat([X, transformed_values], axis=1)
    X = X.drop(col, axis=1)

obj_cols = X.select_dtypes("object").columns
X[obj_cols] = X[obj_cols].astype("category")

SEED = 0
SAMPLE_SIZE = 5000

Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, free_raw_data=False)
np.random.seed(SEED)
sample_idx = np.random.choice(Xt.index, size=SAMPLE_SIZE)
Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
ds = lgb.Dataset(Xs, ys)
dv = lgb.Dataset(Xv, yv, free_raw_data=False)

OBJECTIVE = "binary"
METRIC = "binary_logloss"
MAXIMIZE = False
EARLY_STOPPING_ROUNDS = 10
MAX_ROUNDS = 130
REPORT_ROUNDS = 1

params = {
    "objective": OBJECTIVE,
    "metric": METRIC,
    "verbose": -1,
    "num_classes": 1,
    "n_jobs": 6,
}

evals = {}
model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    # early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
    evals_result=evals,
)

plt.clf()
sns.lineplot(y=evals["training"]['binary_logloss'], x=range(1, MAX_ROUNDS+1), label='training')
sns.lineplot(y=evals["valid"]['binary_logloss'], x=range(1, MAX_ROUNDS+1), label='validation')
plt.xlabel('number of boosting rounds')
plt.ylabel('binary logloss')
plt.show()