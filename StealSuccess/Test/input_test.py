import pandas as pd
import xgboost as xgb

# Load trained model
model = xgb.XGBClassifier()
MODEL_PATH = r"C:\Users\nitro\WhoCaresAboutStealDecision\Models\StealSuccess\XGB\JSON\4f_model.json"
model.load_model(MODEL_PATH)

# Load lookup tables
PLAYERS = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\Connection\player_ID.csv"
players = pd.read_csv(PLAYERS)

SPRINT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\SC\Stack\sprint_speeds.csv"
sprint = pd.read_csv(SPRINT)

POP = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\SC\Stack\catcher_pop.csv"
pop = pd.read_csv(POP)

THROWING = r"C:\Users\nitro\WhoCaresAboutStealDecision\Data\CSV\SC\Stack\throwing_t2.csv"
throwing = pd.read_csv(THROWING)

# Utils
def get_player_id(name):
    row = players.loc[players["PLAYERNAME"] == name]
    if row.empty:
        raise ValueError(f"Player not found: {name}")
    return int(row["MLBID"].iloc[0])

def get_runner_stats(runner_id, year_r):
    row = sprint.loc[(sprint["player_id"] == runner_id) & (sprint["year"] == year_r)]
    if row.empty:
        raise ValueError(f"No sprint speed for runner_id {runner_id} in year {year_r}")
    return {
        "runner_sprint_speed": float(row["sprint_speed"].iloc[0])
    }

def get_catcher_stats(catcher_id, year_c):
    row_throw = throwing.loc[(throwing["player_id"] == catcher_id) & (throwing["year"] == year_c)]

    if row_throw.empty:
        raise ValueError(f"No pop/exchange/arm data for catcher_id {catcher_id} in year {year_c}")

    return {
        "pop_time": float(row_throw["pop_time"].iloc[0]),
        "exchange_time": float(row_throw["exchange_time"].iloc[0]),
        "arm_strength": float(row_throw["arm_strength"].iloc[0])
    }

# Main
def predict_safe(runner_name, r_year, catcher_name, c_year):
    # Convert names
    runner_id = get_player_id(runner_name)
    catcher_id = get_player_id(catcher_name)

    # Get stats
    runner = get_runner_stats(runner_id, r_year)
    catcher = get_catcher_stats(catcher_id, c_year)

    # Build feature vector
    features = {
        "runner_sprint_speed": runner["runner_sprint_speed"],
        "pop_time": catcher["pop_time"],
        "exchange_time": catcher["exchange_time"],
        "arm_strength": catcher["arm_strength"]
    }

    X_input = pd.DataFrame([features])

    # Predict safe probability
    prob_safe = model.predict_proba(X_input)[0][1]
    return prob_safe

prob = predict_safe("Junior Caminero", 2024, "Patrick Bailey", 2025)
print(f"SAFE probability: {prob:.3f}")
