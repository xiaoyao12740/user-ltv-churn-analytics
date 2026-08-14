import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src.features.build_snapshots import feature_columns
from src.paths import METRICS, MODELS, PROCESSED


def _metrics(y, pred):
    return {"mae": float(mean_absolute_error(y,pred)), "rmse": float(mean_squared_error(y,pred) ** .5), "r2": float(r2_score(y,pred))}


def train_ltv(snapshots, seed=42):
    features = feature_columns(snapshots)
    train = snapshots[snapshots.split == "train"]
    val = snapshots[snapshots.split == "validation"]
    test = snapshots[snapshots.split == "test"]
    linear = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
    linear.fit(train[features], np.log1p(train.future_90d_revenue))
    val_linear = np.maximum(0, np.expm1(linear.predict(val[features])))
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=120, min_samples_leaf=8, max_features=.75, n_jobs=-1, random_state=seed))])
    rf.fit(train[features], np.log1p(train.future_90d_revenue))
    val_rf = np.maximum(0, np.expm1(rf.predict(val[features])))
    # Model selection uses validation only. Test remains untouched until this point.
    chosen_name = min({"linear_regression": _metrics(val.future_90d_revenue,val_linear)["mae"], "random_forest": _metrics(val.future_90d_revenue,val_rf)["mae"]}, key=lambda k: {"linear_regression": _metrics(val.future_90d_revenue,val_linear)["mae"], "random_forest": _metrics(val.future_90d_revenue,val_rf)["mae"]}[k])
    preds = {
        "linear_regression": np.maximum(0, np.expm1(linear.predict(test[features]))),
        "random_forest": np.maximum(0, np.expm1(rf.predict(test[features])))
    }
    metrics = {name: _metrics(test.future_90d_revenue, pred) for name,pred in preds.items()}
    metrics["selected_model"] = chosen_name
    chosen_model = linear if chosen_name == "linear_regression" else rf
    train_predictions = np.maximum(0, np.expm1(chosen_model.predict(train[features])))
    metrics["segmentation_value_threshold"] = float(np.quantile(train_predictions, .75))
    METRICS.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True); PROCESSED.mkdir(parents=True,exist_ok=True)
    (METRICS / "ltv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(linear, MODELS / "ltv_linear.joblib"); joblib.dump(rf, MODELS / "ltv_random_forest.joblib")
    out = test[["user_id","snapshot_date","future_90d_revenue"]].copy()
    out["linear_prediction"] = preds["linear_regression"]
    out["predicted_ltv"] = preds[chosen_name]
    out.to_csv(PROCESSED / "ltv_predictions.csv", index=False)
    return metrics, out, rf.named_steps["model"].feature_importances_, features
