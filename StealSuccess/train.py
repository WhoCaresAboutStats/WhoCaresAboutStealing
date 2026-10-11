import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

# Load dataset
CSV_PATH = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\SC\Processed\Output\sb_total.csv"
df = pd.read_csv(CSV_PATH)

# Drop ID columns
drop_cols = [
    "runner_id",
    "pitcher_id",
    "catcher_id",
    "game_pk",
    "at_bat_number",
    "game_date"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

if "inning_topbot" in df.columns:
    df["inning_topbot"] = df["inning_topbot"].astype("category")

# Target
y = df["successful_sb"]
X = df.drop(columns=["successful_sb"])

# Split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DMatrix
dtrain = xgb.QuantileDMatrix(
    X_train, y_train,
    enable_categorical=True
)

dvalid = xgb.QuantileDMatrix(
    X_valid, y_valid,
    enable_categorical=True,
    ref=dtrain
)

params = {
    "objective": "binary:logistic",
    "device": "cuda",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.05,
    "eval_metric": "logloss",
    "enable_categorical": True
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=50
)

# Save model
JSON_PATH = r"C:\Users\nitro\WhoCaresAboutStealDecision\Models\StealSuccess\XGB\JSON"
FULL_PATH = os.path.join(JSON_PATH, "model_2.json")
model.save_model(FULL_PATH)
print(r"Model saved as JSON\model_2.json")
