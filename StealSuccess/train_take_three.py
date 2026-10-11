import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sys


# Load combined dataset
CSV_PATH = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\SC\Processed\Output\Core\sbt_all_tt.csv"
df = pd.read_csv(CSV_PATH)
print(len(df))

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = [
    "runner_sprint_speed",
    "lead_dist_gained_att",
    "pop_time",
    "exchange_time",
    "arm_strength",
    "pitch_tempo"
]

X = df[features]
y = df["successful_sb"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    max_depth=4,
    learning_rate=0.03,
    n_estimators=300,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=0.5,
    eval_metric="logloss"
)


model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

MODEL_PATH = r"C:\Users\nitro\WhoCaresAboutStealDecision\Models\StealSuccess\XGB\JSON\6f_model.json"
model.save_model(MODEL_PATH)
