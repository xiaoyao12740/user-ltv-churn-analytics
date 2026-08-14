import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from src.features.build_snapshots import feature_columns
from src.paths import METRICS, MODELS, PROCESSED


def _metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** .5),
            "r2": float(r2_score(y, pred))}


def _regressors(seed):
    return {
        "linear_regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(
            n_estimators=120, min_samples_leaf=8, max_features=.75, n_jobs=-1, random_state=seed))])}


def _fit_hurdle(frame, features, seed):
    classifier = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(
        n_estimators=120, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=seed))])
    positive = frame.future_90d_revenue > 0
    classifier.fit(frame[features], positive)
    amount = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(
        n_estimators=120, min_samples_leaf=6, max_features=.75, n_jobs=-1, random_state=seed))])
    amount.fit(frame.loc[positive, features], np.log1p(frame.loc[positive, "future_90d_revenue"]))
    return classifier, amount


def _hurdle_predict(models, frame, features):
    classifier, amount = models
    probability = classifier.predict_proba(frame[features])[:, 1]
    conditional = np.maximum(0, np.expm1(amount.predict(frame[features])))
    return probability * conditional


def train_ltv(snapshots, seed=42):
    features = feature_columns(snapshots)
    split_col = "ltv_split" if "ltv_split" in snapshots else "split"
    train = snapshots[snapshots[split_col] == "train"]
    val = snapshots[snapshots[split_col] == "validation"]
    test = snapshots[snapshots[split_col] == "test"]
    if test.snapshot_date.nunique() != 1:
        raise ValueError("LTV test must contain exactly one evaluation snapshot")

    models = _regressors(seed)
    for model in models.values():
        model.fit(train[features], np.log1p(train.future_90d_revenue))
    hurdle = _fit_hurdle(train, features, seed)
    val_predictions = {name: np.maximum(0, np.expm1(model.predict(val[features]))) for name, model in models.items()}
    val_predictions["two_stage_hurdle"] = _hurdle_predict(hurdle, val, features)
    selected = min(val_predictions, key=lambda name: _metrics(val.future_90d_revenue, val_predictions[name])["mae"])

    # Once model family is selected on validation, refit all candidates on labels
    # that are mature at the test as-of date (train + validation only).
    development = snapshots[snapshots[split_col].isin(["train", "validation"])]
    models = _regressors(seed)
    for model in models.values():
        model.fit(development[features], np.log1p(development.future_90d_revenue))
    hurdle = _fit_hurdle(development, features, seed)
    predictions = {
        "zero_baseline": np.zeros(len(test)),
        "mean_baseline": np.full(len(test), development.future_90d_revenue.mean()),
        "median_baseline": np.full(len(test), development.future_90d_revenue.median()),
        **{name: np.maximum(0, np.expm1(model.predict(test[features]))) for name, model in models.items()},
        "two_stage_hurdle": _hurdle_predict(hurdle, test, features)}
    metrics = {name: _metrics(test.future_90d_revenue, pred) for name, pred in predictions.items()}
    metrics["selected_model"] = selected
    selected_train_predictions = (np.maximum(0, np.expm1(models[selected].predict(development[features])))
                                  if selected in models else _hurdle_predict(hurdle, development, features))
    metrics["segmentation_value_threshold"] = float(np.quantile(selected_train_predictions, .75))
    metrics["temporal_design"] = {"split_column": split_col,
        "train_snapshots": sorted(development.loc[development[split_col] == "train", "snapshot_date"].astype(str).unique().tolist()),
        "validation_snapshot": str(val.snapshot_date.iloc[0]), "test_snapshot": str(test.snapshot_date.iloc[0]),
        "label_horizon_days": 90}

    METRICS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    (METRICS / "ltv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name, model in models.items(): joblib.dump(model, MODELS / f"ltv_{name}.joblib")
    joblib.dump(hurdle, MODELS / "ltv_two_stage_hurdle.joblib")
    out = test[["user_id", "snapshot_date", "future_90d_revenue"]].copy()
    for name, pred in predictions.items(): out[f"{name}_prediction"] = pred
    out["predicted_ltv"] = predictions[selected]
    out.to_csv(PROCESSED / "ltv_predictions.csv", index=False)
    importance = models["random_forest"].named_steps["model"].feature_importances_
    return metrics, out, importance, features
